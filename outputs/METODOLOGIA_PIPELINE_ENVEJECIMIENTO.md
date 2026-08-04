# Metodología del pipeline dual de envejecimiento facial con difusión latente

## 1. Propósito y alcance

Este documento describe la metodología implementada para transformar una imagen facial hacia una edad objetivo mediante dos modelos de difusión complementarios: una **rama global**, responsable de los cambios de escala facial completa, y una **rama local**, responsable de las señales de envejecimiento específicas de regiones anatómicas. Ambas salidas se combinan mediante una etapa de fusión que conserva la imagen original como ancla estructural.

El objetivo metodológico no es predecir de forma determinista el futuro biológico exacto de una persona. El sistema aprende una transformación de **edad aparente** plausible, condicionada por texto y regularizada para preservar identidad. La salida debe entenderse como una simulación visual de envejecimiento.

La arquitectura responde a una separación de escalas:

- La rama global modela edad aparente, forma general, volumen, cabello, tono y coherencia de cara completa.
- La rama local modela arrugas, pliegues y textura en zonas faciales anotadas.
- La fusión usa la fotografía original como soporte de identidad, incorpora una versión suavizada del cambio global y después inserta los detalles locales.

En forma resumida:

\[
x_g = G_g(x, a_t),
\qquad
\{\hat{x}_z\}_{z=1}^{Z} = G_l(\{x_z\}, \{s_{z,t}\}),
\]

\[
x_c = x + \alpha(\mathbf{p})\,M_f\,\mathcal{B}_{\sigma_g}(x_g-x),
\qquad
x_b = \operatorname{Insertar}(x_c,\{\hat{x}_z,M_z,b_z\}),
\]

\[
x_{final}=R(x_b),
\]

donde \(x\) es la imagen original, \(a_t\) la edad global objetivo, \(x_z\) el recorte de la zona \(z\), \(s_{z,t}\) su score local objetivo, \(M_f\) una máscara facial, \(M_z\) la máscara suave de la zona, \(b_z\) su caja espacial, \(\mathcal{B}\) un desenfoque gaussiano y \(R\) un refinador opcional de baja intensidad. Si el refinador no se usa, \(x_{final}=x_b\).

## 2. Fuentes de supervisión

El entrenamiento utiliza tres contratos de datos que no deben confundirse entre sí.

| Fuente | Unidad de muestreo | Etiqueta principal | Uso metodológico |
|---|---|---|---|
| Local anotada | Un crop de una región facial | Región, caja y score manual de envejecimiento local | Aprender textura y severidad por zona |
| Global no pareada | Una cara completa | Edad aparente estimada y atributos visuales | Aprender el aspecto global condicionado por edad |
| Longitudinal o *ground truth* | Dos fotos de la misma identidad a edades distintas | Identidad, edad fuente y edad objetivo conocidas | Anclar la rama global a endpoints reales de una misma persona |

### 2.1. Datos locales: crops anotados

Cada anotación local enlaza una imagen con una o varias cajas faciales. Tras descartar slots omitidos, cajas ausentes, scores ausentes e imágenes no disponibles, cada caja válida se convierte en una muestra independiente. Las ocho categorías canónicas implementadas son:

1. frente;
2. glabela o entrecejo;
3. patas de gallo;
4. región bajo el ojo u ojeras;
5. surcos nasogenianos;
6. labio superior;
7. comisuras o líneas de marioneta;
8. puente nasal o región interocular.

Cada slot incluye una caja \((x,y,w,h)\), un score manual en escala \([0,100]\), la región anatómica y contexto demográfico opcional. La caja anotada se amplía con contexto específico de la región y se convierte en un crop cuadrado; cuando alcanza un borde de la imagen se desplaza dentro de sus límites en vez de añadir padding. El crop se redimensiona a \(256\times256\) y se normaliza a \([-1,1]\).

El score usado para describir los píxeles reales se normaliza como

\[
s_z = \frac{s_z^{(100)}}{100}\in[0,1].
\]

El prompt fuente contiene la región y el score observado. Esta correspondencia es deliberada: el prompt que acompaña al crop real debe describir su severidad real y no una versión perturbada. Durante entrenamiento se construye aparte un score objetivo, normalmente orientado hacia envejecimiento, anclaje o un contraste leve.

El split local se realiza por `image_id`, no por crop. Por tanto, los recortes de una misma imagen no pueden repartirse entre entrenamiento y validación. Un muestreador ponderado compensa el desbalance por región y aumenta la frecuencia de ejemplos con señales altas de envejecimiento. Las transformaciones son suaves para no destruir la textura que se pretende aprender.

El batch local principal contiene:

```text
pixel_values   [B,3,256,256]  crop normalizado
score          [B]            score observado normalizado
score_raw      [B]            score en escala 0–100
prompt         List[str]      región + score + contexto
zone_prompt    List[str]      condición anatómica sin score explícito
region_key     List[str]
bbox_crop      [B,4]
image_id       List[str]
```

### 2.2. Datos globales: caras completas con pseudo-etiqueta

La rama global usa imágenes faciales completas a \(512\times512\). A cada imagen se le asocia una estimación automática de edad aparente (`age_pred`) y, cuando la confianza es suficiente, atributos como género aparente, tono de piel, cabello y gafas. Con ellos se forma un prompt descriptivo controlado.

La edad de esta fuente es una **pseudo-etiqueta de edad aparente**, no una edad biológica verificada. Por eso esta rama aprende principalmente la distribución transversal de apariencia por edad:

\[
p(x\mid a, c),
\]

donde \(c\) representa los demás atributos del prompt. Por sí sola, esta fuente no observa cómo cambia la misma identidad con el paso del tiempo.

El batch global contiene como mínimo:

```text
pixel_values   [B,3,512,512]  cara completa normalizada
age            [B]            edad aparente estimada
age_norm       [B]            edad / 100
prompt         List[str]      prompt observado de la imagen
```

### 2.3. Datos longitudinales: pares reales de la misma identidad

La supervisión longitudinal opcional usa FG-NET o AgeDB. Se extraen la identidad y la edad a partir de la convención oficial de nombres/estructura. Para cada identidad se forman únicamente pares hacia adelante:

\[
(x_s,a_s,x_t,a_t), \qquad a_s<a_t,
\]

sujetos a una brecha mínima y máxima de edad. Se limita el número de pares por identidad para impedir que personas con muchas fotografías dominen el entrenamiento. El split se hace por identidad, de modo que una persona completa pertenece a entrenamiento o validación, nunca a ambos.

Los pares son *ground truth* con respecto a **identidad compartida y edades de los endpoints**, pero no con respecto a correspondencia píxel a píxel. Entre las dos fotos pueden variar pose, iluminación, cámara, recorte, expresión y fondo. En consecuencia:

- no se usa una pérdida L1 entre las fotografías;
- no se afirma que \(x_t-x_s\) sea un mapa espacial limpio de envejecimiento;
- el término de delta latente permanece desactivado en la configuración base;
- la señal fiable es el denoising de cada fotografía real condicionado por su edad conocida.

La integración ejecutable de alto nivel conecta actualmente estos pares a la **rama global**. Aunque el wrapper posee una interfaz genérica para una pérdida pareada local, el pipeline actual no extrae ni alinea crops longitudinales y no debe describirse como si entrenara ambas ramas con FG-NET/AgeDB.

## 3. Backbone de difusión y parámetros entrenables

Cada rama es un bundle independiente de difusión latente compuesto por:

- VAE para codificar y decodificar imágenes;
- tokenizer y codificador de texto CLIP;
- UNet condicionado por texto;
- scheduler de ruido para entrenamiento y scheduler para inferencia.

VAE, codificador de texto y pesos base del UNet permanecen congelados. Solo se optimizan adaptadores de bajo rango insertados en proyecciones de atención del UNet (`to_q`, `to_k`, `to_v`, `to_out.0`). La rama global usa LoRA y la rama local usa DoRA. Esta elección permite especializar dos tareas distintas sin ajustar todos los pesos del backbone.

Para una capa lineal, LoRA representa la actualización como

\[
W_{eff}=W_0+\frac{\alpha}{r}BA,
\]

con rango \(r\ll\min(d_{in},d_{out})\). DoRA añade una descomposición de magnitud y dirección del peso, útil para cambios locales finos. Los parámetros entrenables de los adaptadores se conservan en `float32`, aunque el forward use precisión mixta.

## 4. Condicionamiento textual y taxonomía de prompts

### 4.1. Cómo entra el texto al modelo

El prompt no se representa mediante variables tabulares concatenadas al latente. Cada rama usa el **tokenizer y el text encoder CLIP pertenecientes a su propio checkpoint de Stable Diffusion**. Esto mantiene compatibilidad con la distribución de embeddings esperada por el cross-attention del UNet. No se sustituye por un CLIP externo ni se ajustan sus pesos.

Para un prompt \(p\), el codificador congelado produce

\[
h=T_{CLIP}(p)\in\mathbb{R}^{L\times d_c}.
\]

En una capa de atención del UNet, las características visuales producen queries y el texto produce keys y values:

\[
Q=W_Qf(z_t),\qquad K=W_Kh,\qquad V=W_Vh,
\]

\[
\operatorname{Attn}(z_t,h)=
\operatorname{softmax}\left(\frac{QK^\top}{\sqrt d}\right)V.
\]

LoRA global y DoRA local se insertan precisamente en las proyecciones `to_q`, `to_k`, `to_v` y `to_out.0`. Por eso, aunque CLIP esté congelado, los adaptadores aprenden **cómo interpretar y aplicar espacialmente** sus tokens de edad, región, score y atributos visuales. LoRA/DoRA no son embeddings de edad separados ni modifican el vocabulario; especializan la respuesta del UNet a la condición textual.

La construcción de prompts sigue cuatro reglas:

1. La condición de reconstrucción debe describir los píxeles reales observados.
2. La condición objetivo debe cambiar la variable que se quiere editar —edad o score— y conservar atributos estables confiables.
3. Los atributos inferidos solo se incluyen cuando superan un umbral de confianza.
4. Los prompts locales omiten características globales que no pueden verificarse dentro de un crop.

### 4.2. Inventario de prompts por fuente y función

| Contexto | Tipo de prompt | Plantilla conceptual | Uso |
|---|---|---|---|
| Global transversal | Fuente condicionado | `a portrait photo of a {age}-year-old {person}, {trusted attributes}` | `L_diff` sobre la cara real |
| Global transversal | Neutral | Prompt fuente sin edad ni atributos correlacionados con edad | Dropout de condición y doble prompt |
| Global transversal | Objetivo | `a portrait photo of {age-aware person} as {target_age}-year-old, {stable attributes}` | Forward semántico e inferencia global |
| Local anotado | Fuente completo | `a tightly cropped... {region}... aging score of {score}%... {demographic context}` | `L_full` sobre el crop real |
| Local anotado | Zona | `a tightly cropped... {region}, showing facial skin texture` | `L_zone` anatómica |
| Local anotado | Neutral | Prompt local sin score ni semántica explícita de envejecimiento | Dropout de condición y doble prompt |
| Local anotado | Objetivo | Prompt local fuente con descriptor y score objetivo reemplazados | `L_score`, direccional e inferencia local |
| Longitudinal | Endpoint fuente | `a portrait photo of a {source_age}-year-old {person}` | Denoising del endpoint temprano |
| Longitudinal | Endpoint objetivo | `a portrait photo of a {target_age}-year-old {person}` | Denoising principal del endpoint tardío |
| Inferencia | Negativo global/local | Lista de defectos o cambios no deseados | CFG negativo opcional en img2img |
| Refinador | Armonización positivo | Misma persona + envejecimiento natural + textura/luz consistente | Armonizar la fusión |
| Refinador | Armonización negativo | Cambio de identidad + deformaciones + piel plástica + blur | Evitar que el refinador borre o sustituya el resultado |

### 4.3. Prompts globales y atributos adicionales para CLIP

El prompt global observado parte de una edad aparente redondeada y selecciona `man`, `woman` o el fallback conservador `person`. Después puede añadir:

- tono de piel;
- característica de cabello;
- presencia de gafas.

Estos campos solo se incorporan cuando sus predictores superan umbrales de confianza. Una forma real del prompt es:

```text
a portrait photo of a 46-year-old man, medium skin tone, black hair, wearing glasses
```

La finalidad de las características extra es dar a CLIP una descripción más informada de aquello que debe permanecer estable. No deben usarse como sustitutos de la edad ni añadirse indiscriminadamente: cabello gris, piel arrugada, `elderly` u otros atributos fuertemente correlacionados con edad se eliminan del prompt neutral porque permitirían aprender atajos.

Al construir el objetivo global, el código conserva la cola de atributos del prompt fuente y sustituye el concepto de edad/persona. El sustantivo se adapta a la edad para que el texto permanezca semánticamente natural:

| Edad objetivo | Token de persona implementado |
|---|---|
| menor de 5 | `baby` |
| 5–14 | `boy`, `girl` o `child` |
| 15–44 | `man`, `woman` o `person` |
| 45–59 | `middle-aged ...` |
| 60–69 | `older ...` |
| 70 o más | `elderly ...` |

Ejemplo de condición objetivo:

```text
a portrait photo of an elderly man as 75-year-old, medium skin tone, wearing glasses
```

El prompt neutral se deriva eliminando edad explícita y términos como cabello gris, arrugas, `older`, `elderly` o `middle-aged`. Conserva contenido no etario útil. Así, la diferencia entre condición completa y neutral concentra mejor la variable edad.

### 4.4. Prompts locales: región, score y contexto observable

La plantilla fuente local indica expresamente que la entrada es un recorte estrecho y centrado. Esto reduce el prior de generar una cara completa dentro del crop:

```text
a tightly cropped, centered close-up of the {region},
showing facial skin texture and local aging details,
with an aging score of {score}%, for a {demographic context}
```

El contexto demográfico se limpia antes de entrar al prompt. Se eliminan edad global, cabello y género porque un parche pequeño de piel no suele contener evidencia suficiente. Solo se conserva, cuando existe, una categoría amplia como `Asian person`, `white person` o `African American person`; de lo contrario se usa `person`. Esta decisión evita pedirle al DoRA local que reproduzca información invisible en sus píxeles.

El `zone_prompt` elimina score y contexto:

```text
a tightly cropped, centered close-up of the crow's feet region,
showing facial skin texture
```

El prompt objetivo reemplaza el score observado por uno muestreado y añade un descriptor coherente con el modo:

- `aging`: `pronounced`, `strong` o `severe local aging score` según severidad;
- `anchor`: `stable ... local aging score`;
- `contrast`: `reduced but still realistic` o `milder local aging score`.

De esta manera el número no es el único token que transmite intensidad. Para la pérdida direccional se construyen dos versiones del mismo prompt —score alto y score bajo— y se comparte el ruido, aislando en lo posible el efecto de la condición textual.

### 4.5. Prompts longitudinales

FG-NET/AgeDB emplean plantillas deliberadamente simples:

```text
a portrait photo of a 24-year-old woman
a portrait photo of a 61-year-old woman
```

La única información segura es la edad conocida, el género si está disponible y que se trata de la misma identidad según el dataset. No se introduce un token de identidad aprendido ni se copian atributos de una foto a la otra, pues cabello, gafas y estilo pueden cambiar legítimamente entre endpoints.

### 4.6. Prompts negativos de generación

Las ramas global y local admiten un prompt negativo opcional durante img2img. El YAML base lo deja vacío; los notebooks de entrenamiento/monitoreo usan vocabulario de seguridad visual según la escala:

- Global: deformaciones de cara/ojos, apariencia de cadáver o zombi, enfermedad, artefactos, piel plástica y cambios extremos.
- Local: blur, piel excesivamente lisa o plástica, textura distorsionada y artefactos.

El negativo no participa en las pérdidas descritas arriba; es un control de sampling mediante CFG. Debe ser breve y específico para no competir con señales positivas de envejecimiento realista.

### 4.7. Prompt del refinador

El refinador recibe la imagen ya fusionada y un prompt de **armonización**, no un nuevo mandato de envejecimiento. La condición positiva por defecto contiene:

```text
ultra-realistic portrait photo of the same person,
natural facial aging, consistent skin texture, seamless blending,
realistic wrinkles, identity-preserving face, natural lighting
```

La condición negativa por defecto contiene:

```text
changed identity, different person, deformed face, distorted eyes,
distorted mouth, plastic skin, waxy skin, blurry, artifacts,
erased wrinkles, over-smoothed skin, unrealistic texture
```

Cada grupo cumple una función explícita:

- `same person` e `identity-preserving` restringen deriva de identidad;
- `consistent skin texture`, `seamless blending` y `natural lighting` atacan discontinuidades de crops;
- `natural facial aging` y `realistic wrinkles` evitan que armonizar signifique rejuvenecer;
- `erased wrinkles` y `over-smoothed skin` en el negativo protegen el detalle generado por DoRA;
- deformaciones, blur y piel plástica controlan defectos típicos de img2img.

El pipeline permite sobrescribir ambos prompts. Para informar todavía más al CLIP del refinador se puede añadir la edad objetivo y atributos estables de alta confianza:

```text
ultra-realistic portrait photo of the same 75-year-old woman,
wearing glasses, natural facial aging, consistent skin texture,
seamless blending, realistic wrinkles, natural lighting
```

Esta extensión debe mantener el énfasis en armonización. No conviene añadir detalles no observados ni repetir una lista larga de señales de envejecimiento, porque con fuerza img2img excesiva el refinador podría rehacer la cara. En la implementación actual el refinador está congelado, usa fuerza baja y puede disponer de uno o dos text encoders según el pipeline cargado; los prompts se procesan por la interfaz estándar del modelo y no mediante embeddings concatenados manualmente.

### 4.8. Relación con FADING y SelfAge

El diseño toma ideas concretas de dos referencias incluidas en el repositorio:

**FADING — Chen y Lathuilière (2023).** Propone especializar un LDM con edad numérica y un esquema de doble prompt: una condición etaria (`photo of a [age] year old person`) y otra agnóstica a edad (`photo of a person`). También enriquece los prompts reemplazando `person` por `man`/`woman` y por `boy`/`girl` en edades jóvenes. Sus ablaciones atribuyen al doble prompt mejor disentanglement y preservación de atributos no etarios.

**SelfAge — Ito, Endo y Kanamori (2025).** Refina la representación como `{age}-year-old`, incorpora la edad de las imágenes de autorreferencia para separar identidad y edad, cambia el sustantivo según edad extrema (`baby`, `boy/girl`, `man/woman`, `elderly`) y usa LoRA para reducir sobreajuste durante personalización.

Este proyecto adapta esas ideas de la siguiente manera:

| Idea de las referencias | Adaptación implementada aquí |
|---|---|
| Edad numérica explícita | Edad entera en prompts globales y longitudinales |
| Representación `{age}-year-old` | Usada en fuente, objetivo y pares reales |
| Double prompt etario/agnóstico | Prompt dropout y doble forward secuencial ocasional |
| Sustantivo dependiente de edad/género | Función con `baby`, `child`, `middle-aged`, `older`, `elderly`, etc. |
| Atributos para mejorar el targeting de CLIP | Género, tono de piel, cabello y gafas filtrados por confianza |
| LoRA contra sobreajuste | LoRA global; DoRA local para control fino de textura |
| Separar identidad y edad | Pérdida FaceNet, prompt fuente/objetivo y original como ancla de fusión |

No se implementan literalmente Null-text Inversion, reemplazo de mapas de atención de Prompt-to-Prompt ni el token personalizado `⟨token⟩` de SelfAge. La preservación estructural se resuelve aquí mediante img2img de baja fuerza, pérdidas auxiliares y fusión residual global-local. Esta distinción debe conservarse al redactar el paper.

## 5. Forward común de difusión

Las dos ramas comparten el mismo principio de denoising. Para una imagen o crop real \(x_0\), el VAE congelado obtiene el latente

\[
z_0 = c_{vae}\,E(x_0),
\]

usando la media de la distribución latente. Se muestrean un timestep \(t\) y ruido gaussiano \(\epsilon\sim\mathcal{N}(0,I)\). El scheduler construye

\[
z_t=\sqrt{\bar\alpha_t}z_0+\sqrt{1-\bar\alpha_t}\epsilon.
\]

El texto \(p\) se tokeniza y codifica como \(h=T(p)\). El UNet adaptado predice el ruido:

\[
\hat\epsilon_\theta=\epsilon_\theta(z_t,t,h).
\]

Cuando una pérdida auxiliar necesita una imagen editada se estima el latente limpio de un paso:

\[
\hat z_0=\frac{z_t-\sqrt{1-\bar\alpha_t}\hat\epsilon_\theta}
{\sqrt{\bar\alpha_t}},
\qquad
\hat x_0=D(\hat z_0/c_{vae}).
\]

El código también admite una trayectoria DDIM corta para producir una imagen más nítida antes de aplicar estimadores congelados. En la configuración base se usa la estimación de un paso; DDIM corto es una alternativa metodológica ya implementada.

Para la pérdida de ruido se puede aplicar Min-SNR-\(\gamma\):

\[
\operatorname{SNR}(t)=\frac{\bar\alpha_t}{1-\bar\alpha_t},
\qquad
w_t=\frac{\min(\operatorname{SNR}(t),\gamma)}{\operatorname{SNR}(t)},
\]

\[
\mathcal{L}_{diff}=\mathbb{E}\left[w_t
\left\|\hat\epsilon_\theta-\epsilon\right\|_2^2\right].
\]

Esta ponderación reduce el dominio de timesteps con SNR extrema.

## 6. Rama local

### 6.1. Condicionamiento local

El prompt completo describe un close-up de la región, su señal de envejecimiento y un contexto demográfico conservador. El prompt de zona identifica la anatomía sin exigir la misma precisión de score. Para un objetivo de edición se crea un prompt con score \(s_{z,t}\), mientras el prompt fuente permanece asociado al score observado \(s_{z,s}\).

La construcción de objetivos no equivale a rejuvenecer aleatoriamente cada crop. El muestreo privilegia tres comportamientos: aumentar la señal de edad, anclar alrededor del score observado y usar contraste leve para calibración. Esto enseña tanto reconstrucción de ejemplos reales como control por score.

### 6.2. Pérdida local

La función objetivo implementada es

\[
\mathcal{L}_{local}=
\lambda_{full}\mathcal{L}_{full}+
\lambda_{zone}\mathcal{L}_{zone}+
\lambda_{score}\mathcal{L}_{score}+
\lambda_{cycle}\mathcal{L}_{cycle}+
\lambda_{dir}\mathcal{L}_{dir}.
\]

#### Denoising con prompt completo

\[
\mathcal{L}_{full}=\mathbb{E}\left[w_t
\|\epsilon_\theta(z_t,t,T(p_{full}))-\epsilon\|_2^2\right].
\]

Enseña a reconstruir un crop real condicionado por región, score y contexto. Es el término base de la rama.

#### Denoising con prompt anatómico

\[
\mathcal{L}_{zone}=\mathbb{E}\left[w_t
\|\epsilon_\theta(z_t,t,T(p_{zone}))-\epsilon\|_2^2\right].
\]

Refuerza la interpretación de cada región incluso sin el prompt completo. Si se calculan `full` y `zone` juntos, comparten \(z_t\), ruido y timestep, pero requieren dos forwards del UNet porque cambia el texto.

#### Consistencia con ScoreNet

Se genera un crop con el prompt objetivo, se decodifica y se evalúa con ScoreNet congelado \(S\):

\[
\mathcal{L}_{score}=\operatorname{MSE}(S(\hat{x}_{z,t}),s_{z,t}).
\]

ScoreNet recibe RGB decodificado y no latentes. Sus parámetros están congelados, pero su forward no se ejecuta bajo `no_grad`, porque el gradiente debe atravesar \(S\), el decoder y la predicción del UNet hasta los adaptadores DoRA.

#### Consistencia de ciclo opcional

La aproximación de ciclo hace una edición hacia el prompt objetivo, vuelve a añadir ruido y reconstruye con el prompt fuente:

\[
z_0\xrightarrow{p_t}\hat z_{0,t}
\xrightarrow{p_s}\hat z_{0,rec},
\qquad
\mathcal{L}_{cycle}=\|\hat z_{0,rec}-z_0\|_2^2.
\]

Es costosa por los forwards adicionales y permanece apagada en la configuración base.

#### Ordenamiento direccional opcional

Para reducir dependencia del sesgo absoluto de ScoreNet, puede generarse el mismo crop con ruido compartido y dos prompts, uno de score alto y otro bajo:

\[
\mathcal{L}_{dir}=\max(0,m-[S(\hat x_{hi})-S(\hat x_{lo})]).
\]

Este término enseña monotonicidad del control local. Está implementado, pero desactivado en el baseline.

### 6.3. Muestreo de componentes locales

No se evalúan todos los términos por cada batch. El loop elige estocásticamente entre los modos `full`, `score` y `zone`; los modos costosos opcionales se activan mediante flags independientes. El total de la iteración solo incluye los componentes calculados. Este muestreo reduce memoria y tiempo sin cambiar la definición conceptual de la función objetivo esperada.

## 7. Rama global

### 7.1. Condicionamiento y construcción de la edad objetivo

El prompt fuente describe la cara observada con su edad aparente. Para la rama semántica se muestrea una edad objetivo y se crea un prompt nuevo. La distribución favorece envejecimiento hacia adelante, incorpora casos de anclaje alrededor de la edad fuente y una proporción pequeña de rejuvenecimiento leve como regularización.

Esta distinción es esencial:

- en modo de denoising, el prompt fuente debe describir la imagen real;
- en modo semántico, el prompt objetivo describe la transformación que se desea inducir.

### 7.2. Pérdida global

La función objetivo es

\[
\mathcal{L}_{global}=
\lambda_{diff}\mathcal{L}_{diff}+
\lambda_{id}\mathcal{L}_{id}+
\lambda_{age}\mathcal{L}_{age}+
\lambda_{\Delta age}\mathcal{L}_{\Delta age}+
\lambda_{perc}\mathcal{L}_{perc}.
\]

#### Denoising global

`L_diff` es la predicción de ruido sobre la cara real con el prompt fuente y Min-SNR opcional. Enseña la distribución de imágenes faciales reales condicionadas por edad y atributos.

#### Forward semántico

Se parte del latente de la imagen fuente, se genera \(\hat x_t\) con el prompt de edad objetivo y se evalúa mediante modelos auxiliares congelados. En el modo de un paso, el código reconstruye además una referencia fuente con **el mismo ruido y timestep**, usando el prompt fuente. Así, las comparaciones semánticas relativas no castigan principalmente el blur introducido por VAE/estimación de un paso. En modo DDIM corto, la referencia puede ser la imagen limpia.

Los timesteps semánticos se limitan a una ventana de ruido menor que la de denoising completo y cada muestra se pondera por

\[
q_t=\bar\alpha_t^{\gamma_t},
\]

de modo que los estimadores auxiliares tengan mayor influencia cuando la imagen reconstruida es interpretable.

#### Preservación de identidad

Con un encoder facial congelado \(F\):

\[
\mathcal{L}_{id}=1-cos(F(x_{ref}),F(\hat x_t)).
\]

El objetivo es evitar que el cambio de edad sustituya la identidad.

#### Edad absoluta

Con el estimador de edad congelado \(A\):

\[
\mathcal{L}_{age}=\frac{|A(\hat x_t)-a_t|}{c_a},
\]

donde \(c_a\) normaliza la escala de años.

#### Cambio de edad

El cambio predicho se compara con la brecha cronológica solicitada:

\[
\Delta \hat a=A(\hat x_t)-A(x_{ref}),
\qquad
\Delta a_t=a_t-a_s,
\]

\[
\mathcal{L}_{\Delta age}=
\frac{|\Delta\hat a-\Delta a_t|}{c_a}.
\]

Usar \(a_t-a_s\) mantiene este objetivo distinto de la edad absoluta y evita contar dos veces el mismo error.

#### Distancia perceptual opcional

\[
\mathcal{L}_{perc}=\operatorname{LPIPS}(x_{ref},\hat x_t).
\]

Está implementada, pero permanece apagada por defecto porque puede penalizar cambios legítimos de edad y aumenta el costo.

### 7.3. Muestreo de modos globales

El entrenamiento alterna entre:

- `diff`: un forward de denoising sin decoder ni estimadores semánticos;
- `semantic`: generación objetivo, decoder y auxiliares de edad/identidad;
- `all`: combinación disponible para depuración o equipos con más memoria.

El loop principal muestrea `diff` o `semantic` en cada microbatch. La separación controla el pico de memoria, especialmente en caras de \(512\times512\).

## 8. Supervisión longitudinal de la rama global

La pérdida pareada se ejecuta como un backward adicional e intermitente dentro del entrenamiento global. No reemplaza la pérdida global transversal y no concatena la foto fuente al UNet.

Para un par real se codifican independientemente ambos endpoints. Se comparte el timestep y el ruido entre ellos para reducir varianza:

\[
z_{s,t}=q(z_s,t,\epsilon),
\qquad
z_{t,t}=q(z_t,t,\epsilon).
\]

El endpoint de edad posterior aporta la señal principal y el endpoint fuente regulariza la distribución:

\[
\mathcal{L}_{pair}=
\lambda_{t}\mathcal{L}_{diff}(x_t\mid a_t)+
\lambda_{s}\mathcal{L}_{diff}(x_s\mid a_s)+
\lambda_{lat}\mathcal{L}_{lat\Delta}.
\]

El término opcional de delta latente compara por coseno el cambio entre estimaciones de \(z_0\) con el delta latente real. Debido a la falta de registro espacial entre fotos, el baseline fija \(\lambda_{lat}=0\). La contribución efectiva a un step global es

\[
\mathcal{L}_{step}=\mathcal{L}_{global}^{(modo)}+
\mathbb{1}_{pair}\,\omega_{pair}\mathcal{L}_{pair},
\]

donde \(\mathbb{1}_{pair}\) indica los pasos configurados para supervisión longitudinal y \(\omega_{pair}\) controla su escala.

Esta supervisión ayuda a que el adaptador global observe edades reales de una misma identidad, pero no convierte el problema en traducción supervisada píxel a píxel.

## 9. Prompts neutrales, dropout y doble prompt

Ambas ramas construyen una condición fuente, una condición neutral y una condición objetivo.

- El prompt neutral global elimina edad explícita y atributos fuertemente correlacionados con edad.
- El prompt neutral local elimina el score y la frase de severidad, pero conserva la región.
- Con una probabilidad pequeña, el prompt fuente se reemplaza por el neutral. Es una forma de dropout de condición inspirada en classifier-free guidance.
- En algunos steps de denoising se ejecuta un entrenamiento explícito de doble prompt: un forward/backward con prompt fuente y, después de liberar ese grafo, otro con prompt neutral.

Los dos prompts no se concatenan ni requieren una cross-attention nueva. El doble forward solo aparece ocasionalmente y de forma secuencial para no mantener dos grafos grandes en memoria. El modo semántico usa directamente el prompt objetivo.

## 10. Entrenamiento conjunto por etapas

Las ramas comparten un wrapper de entrenamiento, pero no se actualizan simultáneamente en el mismo grafo. Dentro de cada época se recorren en un orden configurable, normalmente local y luego global:

```text
época e
  ├── activar bundle local y auxiliares locales
  │     ├── muestrear modo local
  │     ├── backward base
  │     ├── backward fused opcional
  │     ├── backward pareado local (interfaz disponible, no cableada en el CLI actual)
  │     └── actualizar solo DoRA local
  ├── descargar rama local a CPU
  ├── activar bundle global y auxiliares globales
  │     ├── muestrear diff o semantic
  │     ├── backward base
  │     ├── backward longitudinal opcional
  │     └── actualizar solo LoRA global
  └── descargar rama global a CPU y guardar checkpoints/muestras
```

Se emplean acumulación de gradiente, clipping, precisión mixta, gradient checkpointing y offload de la rama inactiva. Los schedulers incluyen warmup y decaimiento hasta una tasa mínima. Los checkpoints se guardan por rama; las copias de inferencia contienen los pesos del adaptador y metadatos necesarios para reconstruir su arquitectura.

Los valores concretos pueden variar entre experimentos. La configuración base usa más épocas para la rama local que para la global, una tasa de aprendizaje local ligeramente mayor y muestreo más frecuente de los términos de denoising que de los auxiliares costosos.

## 11. Pérdida de fusión local opcional durante entrenamiento

Además del loader aleatorio de crops existe un loader alineado por imagen. Cada item agrupa una cara completa y todos sus recortes válidos:

```text
full_pixel_values  [B,3,512,512]
pixel_values       [B,K,3,256,256]
boxes              [B,K,4]
masks              [B,K,1,256,256]
target_scores      [B,K]
valid_mask         [B,K]
```

La ruta genera los \(K\) crops objetivo con gradiente hacia DoRA y los reinserta mediante operaciones tensoriales diferenciables. La salida global usada como contexto se desacopla del grafo, por lo que esta pérdida actualiza únicamente la rama local.

Se definen dos términos:

1. **Persistencia del score después de la fusión.** Se vuelven a extraer las regiones de la cara fusionada y ScoreNet verifica que el detalle local sobreviva a la reinserción:

\[
\mathcal{L}_{fuse-score}=\operatorname{MSE}
(S(\operatorname{Crop}(x_{fused})),s_t).
\]

2. **Costura.** Se construye una banda alrededor de los bordes de las máscaras y se penaliza el cambio introducido allí:

\[
\mathcal{L}_{seam}=\operatorname{mean}
\left[M_{edge}\odot|x_{fused}-x_{coarse}|\right].
\]

La pérdida es

\[
\mathcal{L}_{fused}=
\lambda_{fs}\mathcal{L}_{fuse-score}+
\lambda_{seam}\mathcal{L}_{seam}.
\]

Este mecanismo está implementado pero desactivado por defecto. Es distinto de la fusión de inferencia: la versión de entrenamiento prioriza diferenciabilidad y simplicidad, mientras la de inferencia añade máscara espacial de confianza global y ajuste de color.

## 12. Forward completo de inferencia

### 12.1. Generación global y local

La imagen original se procesa mediante img2img global con el prompt de edad objetivo para obtener \(x_g\). En paralelo conceptual —aunque las ramas pueden cargarse secuencialmente por memoria— se extraen los crops definidos por región y se editan mediante img2img local con sus scores objetivo. Cada salida local conserva su crop generado, caja y máscara.

La rama local puede reciclar su salida durante más de un pase img2img. Esto es una opción de sampling para intensificar o estabilizar detalle, no una tercera rama entrenada.

### 12.2. Residual global de baja frecuencia

La imagen global completa no sustituye directamente a la original. Se calcula

\[
r=x_g-x,
\qquad
r_{low}=\mathcal{B}_{\sigma_g}(r).
\]

El filtrado reduce cambios de alta frecuencia producidos por la rama global y conserva principalmente dirección global de edad. A partir de todas las máscaras locales se construye su unión suave \(M_L\). El mapa de confianza global es

\[
\alpha(\mathbf p)=
\alpha_{in}M_L(\mathbf p)+
\alpha_{out}[1-M_L(\mathbf p)],
\qquad \alpha_{in}<\alpha_{out}.
\]

Por tanto, la rama global interviene menos dentro de las zonas donde la rama local posee detalle especializado y más en mejillas laterales, mandíbula, mentón y otras áreas no cubiertas por crops. Con máscara facial \(M_f\):

\[
x_c=\operatorname{clip}
\left[x+\alpha(\mathbf p)M_f(\mathbf p)r_{low},0,1\right].
\]

Esta etapa no tiene parámetros entrenables. Sus salidas diagnósticas incluyen el residual crudo, residual suavizado, unión de máscaras y mapa \(\alpha\).

### 12.3. Ajuste cromático de cada crop

Antes de insertar un crop local, se ajustan por canal su media y desviación estándar a las de la región correspondiente de \(x_c\), calculadas preferentemente dentro de la máscara:

\[
\tilde x_z=
\frac{\hat x_z-\mu(\hat x_z)}{\sigma(\hat x_z)+\varepsilon}
\sigma(x_c^{z})+\mu(x_c^{z}).
\]

Con fuerza cromática \(\rho\):

\[
x_z^{match}=(1-\rho)\hat x_z+\rho\tilde x_z.
\]

Esto reduce parches con iluminación o tono incompatibles con la cara base.

### 12.4. Reinserción con feathering

La máscara del crop se redimensiona a la caja, se suaviza y se escala por una intensidad de inserción. Los crops se componen secuencialmente:

\[
x^{(0)}=x_c,
\]

\[
x^{(z)}=M_z^{eff}\odot x_z^{match}+
(1-M_z^{eff})\odot x^{(z-1)}.
\]

El resultado \(x_b=x^{(Z)}\) preserva el contexto global y evita bordes rectangulares duros. En zonas solapadas, el orden secuencial y las máscaras suaves determinan la combinación.

### 12.5. Refinador opcional

La fusión determinista puede entregarse directamente. De forma opcional, \(x_b\) se pasa a un modelo img2img congelado con fuerza baja y un prompt centrado en continuidad de textura, iluminación e identidad:

\[
x_r=R_{img2img}(x_b,p_{harm},\eta_{low}).
\]

El refinador no crea el envejecimiento desde cero ni reemplaza las dos ramas. Su función es armonizar costuras, textura y color. El sistema devuelve por separado `x_blend`/`x_final` y `x_refined`, lo cual permite evaluar si el refinador preserva o borra el detalle local.

## 13. Diferencia entre entrenamiento e inferencia

Durante entrenamiento, las imágenes observadas se corrompen con ruido y el modelo aprende denoising condicionado. Las edades/scores objetivos usados por losses semánticas se construyen a partir de la muestra fuente; no existe una imagen objetivo alineada para cada edición transversal.

Durante inferencia, el usuario fija \(a_t\) y los scores \(s_{z,t}\). Los adaptadores se usan dentro de img2img para producir cambios controlados. Después se realiza la fusión residual y la reinserción local. La supervisión longitudinal modifica cómo se aprendieron los pesos globales, pero no cambia el contrato de inferencia.

## 14. Objetivo total esperado y lectura correcta

Debido al muestreo de modos, no hay un único step que evalúe siempre todas las pérdidas. Conceptualmente, la optimización minimiza la expectativa

\[
\mathbb{E}[\mathcal L]=
\mathbb{E}_{m_l}[\mathcal L_{local}^{(m_l)}]
+\mathbb{E}_{m_g}[\mathcal L_{global}^{(m_g)}]
+\mathbb{E}[\mathbb{1}_{pair}\omega_{pair}\mathcal L_{pair}]
+\mathbb{E}[\mathbb{1}_{fused}\mathcal L_{fused}],
\]

pero cada rama se optimiza en su propio bloque y con su propio optimizador. Las pérdidas pareada y fused son adicionales e intermitentes. Esta formulación evita interpretar el modelo como una única red end-to-end que siempre mantiene simultáneamente los dos UNet y todos los auxiliares en memoria.

## 15. Supuestos, límites y decisiones metodológicas

1. La edad de FFHQ es aparente y estimada; contiene error de etiqueta.
2. Los scores locales son señales ordinales/continuas de severidad visual, no mediciones clínicas universales.
3. Los pares longitudinales comparten identidad y edad conocida, pero no registro espacial.
4. La preservación de identidad depende de un encoder facial auxiliar y de usar la original como ancla de fusión; no es una garantía biométrica.
5. La rama global aprende una dirección poblacional además de la señal longitudinal disponible.
6. La rama local controla únicamente las zonas anotadas; el envejecimiento fuera de ellas proviene del residual global.
7. La fusión de inferencia es principalmente determinista y no se entrena conjuntamente con ambas ramas.
8. El refinador es opcional y debe evaluarse por separado porque podría suavizar señales locales.
9. Variaciones demográficas y de calidad de imagen deben auditarse para evitar que edad, género aparente, tono de piel o estilo fotográfico actúen como atajos espurios.

## 16. Configuración de referencia no exhaustiva

Los hiperparámetros pertenecen a la sección experimental, no al núcleo del método. Como referencia para reproducir la implementación base:

| Elemento | Configuración base |
|---|---|
| Resolución local/global | 256 / 512 |
| Adaptadores | DoRA local, LoRA global |
| Pérdida local base | full + zone + score; cycle y direccional opcionales |
| Pérdida global base | diff + identidad + edad + delta de edad; LPIPS opcional |
| Min-SNR | activo en configuración YAML base |
| Fused loss | implementada, apagada por defecto |
| Pares longitudinales | opcionales, conectados a global |
| Fusión de inferencia | residual global suavizado + color match + máscaras suaves |
| Refinador | opcional, congelado y de baja intensidad |

## 17. Correspondencia con la implementación

Las partes metodológicas anteriores corresponden a los siguientes componentes:

| Componente | Implementación principal |
|---|---|
| Crops y anotaciones locales | `data/local_path_dataset.py` |
| Loader alineado para fusión | `data/local_fused_dataset.py` |
| Caras globales y pseudo-etiquetas | `data/global_path_datasets.py` |
| Pares FG-NET/AgeDB | `data/paired_aging_dataset.py` |
| Construcción de loaders | `data/create_data.py` |
| LoRA/DoRA y bundles | `src/diffusion_pipeline/` |
| Pérdida local | `src/loss/local_loss.py` |
| Pérdida global | `src/loss/global_loss.py` |
| Auxiliares globales | `src/loss/global_aux_bundle.py` |
| Pérdida longitudinal | `src/loss/paired_supervision_loss.py` |
| Pérdida fused | `src/loss/local_fused_loss.py` |
| Construcción de prompts objetivo | `src/training/target_prompt_building.py` |
| Loops por rama | `src/training/train_one_epoch_local.py`, `train_one_epoch_global.py` |
| Wrapper de entrenamiento | `src/training/train_aging_model.py` |
| Fusión global-local | `src/inference/global_local_fusion.py` |
| Operaciones deterministas | `src/inference/deterministic_fusion_ops.py` |
| Refinador opcional | `src/inference/fusion_refiner_helpers.py` |

## 18. Secuencia metodológica completa

```text
Preparación
  1. Asociar imágenes locales con cajas, regiones y scores.
  2. Construir crops cuadrados y split por image_id.
  3. Asociar caras globales con edad aparente y atributos.
  4. Opcional: construir pares longitudinales y split por identidad.

Entrenamiento local
  5. Codificar crop real y prompt fuente.
  6. Muestrear full, zone o score.
  7. Predecir ruido o generar crop objetivo para ScoreNet.
  8. Actualizar únicamente DoRA local.
  9. Opcional: verificar score y costuras tras fusión diferenciable.

Entrenamiento global
 10. Codificar cara real y prompt fuente.
 11. Muestrear denoising o forward semántico hacia edad objetivo.
 12. Aplicar edad, delta de edad e identidad sobre la reconstrucción.
 13. Opcional: añadir denoising de endpoints longitudinales reales.
 14. Actualizar únicamente LoRA global.

Inferencia
 15. Generar cara global a edad objetivo.
 16. Generar crops locales con scores objetivo.
 17. Extraer y suavizar el residual global.
 18. Atenuar global dentro de crops y reforzarlo fuera de ellos.
 19. Ajustar color e insertar cada crop con máscara suave.
 20. Opcional: armonizar la imagen con refinador img2img de baja fuerza.
```

## 19. Referencias metodológicas consultadas

- Xiangyi Chen y Stéphane Lathuilière. *Face Aging via Diffusion-based Editing (FADING)*, 2023, arXiv:2309.11321. Copia local: `references/Cheng - Face Aging via Diffusion-based Editing (2023).pdf`.
- Taishi Ito, Yuki Endo y Yoshihiro Kanamori. *SelfAge: Personalized Facial Age Transformation Using Self-reference Images*, 2025, arXiv:2502.13987. Copia local: `references/difussion 2.pdf`.

Estas referencias sustentan la discusión de especialización del modelo, edad numérica, doble prompt, enriquecimiento semántico, tokens dependientes de la edad y adaptación de bajo rango. Las formulaciones de crops, scores locales, fusión residual espacial, color matching y fused loss corresponden a la implementación específica de este proyecto.

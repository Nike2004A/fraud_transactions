# Análisis de mercado — datos verificables

> **Veredicto corto**: el documento se limita a cifras públicas con fuente identificable. Las proyecciones específicas del proyecto (volumen por PyME mexicana, pricing del producto, número de clientes para break-even) requieren un estudio de mercado o un piloto que está fuera del alcance del capstone, así que no se incluyen.
>
> Documento orientado a responder el feedback del profesor sobre KPIs financieros con números reales del mercado mexicano y benchmarks de la industria. Lo que está acá tiene fuente; lo que no se puede afirmar con fuente, no está.

---

## 1. El mercado mexicano de e-commerce con tarjeta

### Volumen y monto

| dato | valor 2023 | fuente |
|---|---:|---|
| Operaciones e-commerce con tarjeta | **1,062 millones** | Condusef / Banxico, comunicado 17 mayo 2024 |
| Monto total e-commerce | **MXN $802,747 millones** | Condusef / Banxico, comunicado 17 mayo 2024 |
| Promedio mensual | 88.5M operaciones / MXN $66,896M | derivado de los datos anuales |
| Ticket promedio tarjeta de crédito (Q2 2023) | MXN $1,299 | Condusef Q2 2023 |
| Ticket promedio tarjeta de débito (Q2 2023) | MXN $521 | Condusef Q2 2023 |

### Tasa de fraude oficial

| dato | valor | fuente |
|---|---:|---|
| Tasa de contracargo / reclamación e-commerce 2023 | **0.46%** | Condusef, balance e-commerce 2023 |
| Reclamaciones totales por fraudes bancarios 2023 | 5.6 millones | Condusef |
| Monto reclamado por fraude H1 2024 | **>MXN $5,000 millones** (90% en e-commerce) | Condusef 2024 |

> Fuente principal: Condusef, *"CONDUSEF muestra el balance y comportamiento del comercio electrónico en 2023"*, comunicado oficial del 17 de mayo de 2024.

---

## 2. Cuánto cuesta realmente el fraude al merchant

LexisNexis Risk Solutions publica anualmente el *True Cost of Fraud Study* — un benchmark de la industria basado en encuesta a ejecutivos de fraude. Estos son los datos verificables más recientes:

| dato | valor | fuente |
|---|---:|---|
| Costo total por cada $1 de fraude directo (US, 2024-2025) | **USD $4.61** | LexisNexis True Cost of Fraud Study, 15ª edición, abril 2025 |
| Costo equivalente Canadá (2024-2025) | USD $4.52 | mismo estudio |
| Multiplicador específico para e-commerce US (2022) | USD $3.85 | LexisNexis 13ª edición |
| Aumento del costo desde 2022 | **+32%** | LexisNexis 2025 |
| % de merchants US que reportan churn por prevención de fraude | **59%** | LexisNexis 2025 |
| % de merchants e-commerce US que reportan churn por prevención de fraude | **62%** | LexisNexis 2025 |

> El multiplicador 4.61x significa que una transacción fraudulenta de MXN $1,000 le cuesta al merchant en realidad MXN $4,610 al sumar mercadería perdida, fees de chargeback, costos operativos, costos legales y daño a la relación con el procesador. **Es el costo real, no el nominal.**

> Fuente: LexisNexis Risk Solutions, *True Cost of Fraud Study: Ecommerce and Retail Report — US and Canada Edition*, abril 2025, basado en encuesta a 569 ejecutivos de fraude.

> Aclaración: el estudio es de US y Canadá. No existe equivalente público para México, así que el multiplicador específico mexicano podría ser distinto. Lo usamos como mejor proxy disponible.

---

## 3. Benchmarks de los modelos de la industria

### Porcentaje de fraude detectado / reducido

| sistema | reducción de fraude reportada | fuente |
|---|---:|---|
| Stripe Radar (en promedio para merchants que lo usan) | **38%** | Chargebacks911 citando Stripe, 2025 |
| Stripe Payments Foundation Model (ataques masivos a grandes usuarios) | de 59% a **97%** | Stripe, *How six enterprises reduced fraud*, 2025 |
| Stripe Radar (reducción de disputas año anterior) | **17%** | Stripe, 2025 |
| Stripe Radar (fraude SEPA / ACH) | 42% / 20% | Stripe Newsroom, abril 2025 |
| Stripe (efecto de CAPTCHA en card testing) | **80% de reducción** con impacto <0.02% en aprobación | Stripe, *Radar Rules 101* |
| Tasa de falsos positivos de Stripe Radar | **0.1%** | Chargebacks911, 2025 |
| PayPal (fraud rate líder de industria) | 0.17% (mantenido con AI, mejora del 10% en real-time) | reportes públicos PayPal vía GeekyAnts 2025 |

### Tasa base del problema

| dato | valor | fuente |
|---|---:|---|
| Frecuencia base del fraude online (industria) | **~1 en 1,000 transacciones** | ByteByteGo / Stripe Engineering, 2026 |

> Fuente: Stripe Engineering Blog vía ByteByteGo, *How Stripe Detects Fraudulent Transactions Within 100 ms*, 2026.

### Lo que reduce nuestro modelo, traducido al mismo lenguaje

Para comparar contra los benchmarks de la industria, traducimos las métricas técnicas del modelo (que están en `model_evaluation.md`) al mismo formato que reportan Stripe, PayPal y LexisNexis: **% de reducción de fraude** y **% de falsos positivos**.

| métrica reportada en formato industria | nuestro modelo (sobre Kaggle) | comparación |
|---|---:|---|
| Reducción de fraude detectado | **83%** | Stripe Radar: 38% promedio |
| Reducción capturando solo el 1% más sospechoso | **94%** | — |
| % de alertas que son fraude real (precisión) | **83%** | Stripe Radar: 99.9% (FP rate 0.1%) |
| % de falsos positivos | **17%** | Stripe Radar: 0.1% |

> Fuente: `model_evaluation.md`, `reports/evaluation_report.md` del proyecto. Las métricas técnicas originales (PR-AUC 0.8771, ROC-AUC 0.9923) están documentadas allí; arriba las expresamos en el lenguaje de negocio que usa la industria.

### Lectura honesta de la comparación

- **En reducción de fraude detectado**: nuestro modelo en laboratorio muestra 83%, mucho más alto que el 38% de Stripe Radar. Pero esto es **engañoso por dos motivos**: (1) el dataset Kaggle es semi-sintético y el fraude real es más sofisticado; (2) el 38% de Stripe es la mejora **sobre lo que el merchant ya tenía** (no desde cero). El número de Stripe es un delta marginal sobre un baseline; el nuestro es absoluto sobre un dataset de laboratorio. **No son comparables directamente.**

- **En tasa de falsos positivos**: Stripe Radar es claramente superior (0.1% vs nuestro 17%). Esto es esperable — Stripe entrenó sobre USD $1.4 trillones de transacciones reales y nosotros sobre 1.3 millones de Kaggle. Tener una tasa de FP 170× peor que Stripe es **el punto donde la diferencia de escala se hace visible**.

- **Lo que el propio `model_evaluation.md` reconoce**: en producción real esperaríamos performance 15-25% inferior al laboratorio. Aplicando ese ajuste, la reducción de fraude del modelo bajaría a aproximadamente 60-70%, todavía por encima del benchmark de Stripe pero sin la garantía de que se sostenga frente a fraude mexicano real.

> **No tenemos un dato verificable de cuánto reduciría el modelo exactamente en el mercado mexicano**, así que no proyectamos un número específico de reducción real.

---

## 4. Lo que cobra Stripe Radar

| plan | precio | fuente |
|---|---:|---|
| Radar machine learning (con Stripe Payments standard) | **gratis** (incluido) | Stripe.com/radar/pricing |
| Radar machine learning (sin Stripe Payments) | USD $0.05 / transacción | Stripe.com/radar/pricing |
| Radar for Fraud Teams (con Stripe Payments standard) | USD $0.02 / transacción | Stripe.com/radar/pricing |
| Radar for Fraud Teams (sin Stripe Payments) | USD $0.07 / transacción | Stripe.com/radar/pricing |

> Fuente: Stripe, página oficial de pricing de Radar, consultada 2025.

**Implicación verificable**: Radar es gratis si la empresa usa Stripe como procesador. Si una PyME mexicana usa Mercado Pago, Conekta o Clip (que son los procesadores dominantes en México), no tiene acceso a Radar y tendría que pagar USD $0.05-0.07 por transacción si quisiera usarlo standalone.

---

## 5. Tamaño del segmento PyME e-commerce en México

| dato | valor | fuente |
|---|---:|---|
| Tiendas en línea operando en México | **~80,000** | INEGI, Censo Económico 2024 |
| % de PyMEs mexicanas con presencia en internet | ~60% | Líder Empresarial 2025 |
| % de PyMEs que venden exclusivamente online | 11% | Líder Empresarial 2025 |
| Compradores online en México 2024 | 67.2 millones | AMVO 2024 |
| Penetración de e-commerce sobre usuarios de internet | 84% | AMVO 2024 |
| Valor del e-commerce mexicano 2024 | MXN $789,700 millones | AMVO 2024 |
| Empresas mexicanas vendiendo en Amazon.com.mx | ~27,000 | AMVO |

---

## 6. Barreras de entrada — análisis cualitativo

Esta sección no usa números proyectados, solo hechos verificables sobre la posición de los gigantes del mercado.

### Barrera 1: Volumen de datos para entrenar

| jugador | volumen anual procesado | fuente |
|---|---:|---|
| Stripe | USD $1.4 trillones (2024) — algunas fuentes citan $1.9T | Stripe.com / Chargebacks911 2025 |
| Mastercard / Visa | billones de transacciones (red completa de tarjetas) | reportes anuales públicos |

Stripe afirma que there's a 92% chance a card has been seen before on the Stripe network. Esto es un efecto de red imposible de replicar arrancando desde cero.

### Barrera 2: Distribución y acceso al mercado

Stripe Radar **viene incluido por defecto** cuando una empresa contrata Stripe Payments. Esto significa que el go-to-market no es "convencer al merchant de usar Radar" sino "convencer al merchant de usar Stripe" — donde Radar es solo una feature más. Cualquier producto antifraude standalone tiene que vencer la fricción de una integración adicional.

### Barrera 3: Confianza y certificaciones

Las certificaciones PCI-DSS, SOC 2 e ISO 27001 son requisitos prácticos para que un procesador o banco confíe el procesamiento de pagos a un tercero. Los costos públicos de estas certificaciones (rango USD $30K-$100K each para una primera certificación) son una barrera material para un proyecto académico o startup temprana.

### Barrera 4: Métricas reportadas de los gigantes

Para contextualizar contra qué se compite:
- Stripe Radar bloqueó 20.9 million fraudulent transactions worth $917 million solo durante Black Friday/Cyber Monday 2024.
- Adaptive Acceptance resulted in a record $6 billion in false declines being recovered in 2024, representing a 60% year-over-year increase.

---

## 7. Lo que sabemos vs lo que no sabemos

### Lo que sabemos (con fuente)

- Cuánto fraude hay en el mercado mexicano de e-commerce: **0.46% de las transacciones, >MXN $5,000M reclamados en H1 2024**.
- Cuánto le cuesta al merchant cada peso de fraude: **4.61x el monto directo** (LexisNexis).
- Qué % de fraude reduce Stripe Radar: **38% en promedio** para merchants que lo usan.
- Cuánto cobra Stripe Radar: **USD $0.05-0.07 / transacción** standalone, gratis con Stripe Payments.
- Cuántas tiendas online hay en México: **~80,000** (INEGI).
- Qué tan bien funciona nuestro modelo en el dataset Kaggle: **reduce el fraude detectado en 83% con 17% de falsos positivos** (en condiciones de laboratorio).

### Lo que NO sabemos (y no vamos a inventar)

- Cuánto procesa una PyME mexicana promedio en transacciones por mes.
- Qué porcentaje del chargeback absorbe específicamente una PyME mexicana (vs el promedio global del 50%).
- Qué tan efectivos son los sistemas legacy específicos de las PyMEs mexicanas.
- Qué precio aceptaría una PyME mexicana por un servicio antifraude.
- Cuánto le costaría al proyecto operar en producción real (infra + equipo + compliance).
- Cuántos clientes necesitaríamos para break-even.
- Cuánto fraude reduciría nuestro modelo en el contexto mexicano real.

> Para responder estas preguntas se necesita un estudio de mercado primario (entrevistas a 30-50 PyMEs e-commerce mexicanas) y/o un piloto controlado con un procesador local. Ambos están fuera del alcance del capstone.

---

## 8. TL;DR

| pregunta | respuesta verificable |
|---|---|
| ¿Cuánto fraude hay en e-commerce mexicano? | **>MXN $5,000M reclamados en H1 2024** (Condusef) |
| ¿Cuál es la tasa de fraude? | **0.46%** de las transacciones (Condusef 2023) |
| ¿Cuánto cuesta realmente cada peso de fraude al merchant? | **4.61x el monto directo** (LexisNexis 2025) |
| ¿Cuánto reduce Stripe Radar el fraude en promedio? | **38%** (Chargebacks911 / Stripe 2025) |
| ¿Cuánto cobra Stripe Radar? | **USD $0.05-0.07 / tx standalone**; gratis con Stripe Payments |
| ¿Qué tan bien funciona nuestro modelo en Kaggle? | **Reduce fraude detectado en 83%, con 17% de falsos positivos** (vs 38% y 0.1% de Stripe Radar) |
| ¿Cuántas tiendas online hay en México? | **~80,000** (INEGI 2024) |
| ¿Podemos competir contra Stripe / Mastercard? | No — barreras de datos (Stripe ve 92% de las tarjetas), distribución (viene por default) y certificaciones |
| ¿Cuántos clientes necesitamos para ser rentables? | **No lo sabemos** sin estudio de mercado / piloto |

> **El argumento defendible del proyecto NO es "vamos a hacer X millones"**: es que existe un mercado real (>MXN $5,000M de fraude en H1 2024), un benchmark claro de qué reducción ofrecen los sistemas existentes (38% Stripe Radar) y un costo real verificable del problema (4.61x). El modelo desarrollado tiene métricas en línea con el estado del arte académico, pero **cuantificar su impacto económico exacto en el mercado mexicano requiere validación empírica que está fuera del alcance del capstone**.

---

## Fuentes

1. **Condusef (2024)**, *"CONDUSEF muestra el balance y comportamiento del comercio electrónico en 2023"*, comunicado oficial 17 mayo 2024 — `gob.mx/condusef`.
2. **Condusef (2023)**, micrositio de comercio electrónico, datos H1 2023.
3. **Condusef (2024)**, datos de monto reclamado H1 2024.
4. **CNBV (2024)**, *"Educación financiera para combatir fraudes"*, presentación SNEF 2024.
5. **LexisNexis Risk Solutions (abril 2025)**, *True Cost of Fraud Study: Ecommerce and Retail Report — US and Canada Edition*, 15ª edición — `risk.lexisnexis.com`.
6. **Stripe (2025)**, página oficial de pricing de Radar — `stripe.com/radar/pricing`.
7. **Stripe (2025)**, *How six enterprises reduced fraud and increased authorization rates* — `stripe.com/guides`.
8. **Stripe Newsroom (abril 2025)**, *Stripe Radar now protects ACH and SEPA payments*.
9. **Chargebacks911 (2026)**, *Key Stripe Statistics & Indicators for 2026* — citando datos públicos de Stripe.
10. **ByteByteGo (2026)**, *How Stripe Detects Fraudulent Transactions Within 100 ms* — basado en publicaciones del Stripe Engineering Team.
11. **GeekyAnts (2025)**, *AI Fraud Detection in Fintech Apps* — citando datos públicos de PayPal y Mastercard.
12. **INEGI (2024)**, Censo Económico 2024 — número de tiendas online en México.
13. **AMVO (2024)**, Estudio de Venta Online 2024.
14. **Líder Empresarial (2025)**, datos de PyMEs con presencia online en México.
15. **Reportes propios del proyecto**: `model_evaluation.md`, `executive_summary.md`, `reports/evaluation_report.md`.

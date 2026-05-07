# Arquitectura E2E real-time — cómo funciona el sistema en producción

> **Veredicto corto**: el proyecto entrenó un modelo de detección de fraude (Fases 1-4). Este documento describe cómo ese modelo se opera en producción atendiendo transacciones en tiempo real. El sistema procesa cada transacción del checkout de la PyME en menos de 300 ms, devuelve una decisión (aprobar / escalar / bloquear), y los casos escalados llegan a **nuestro analista de fraude interno** — no al cliente PyME, que recibe el resumen quincenal por email. Toda la operación es nuestra; la PyME solo paga el servicio.
>
> Documento orientado a responder el feedback del profesor sobre arquitectura E2E real-time. El flujo sigue el orden exacto que pidió el feedback: **Transaction → Feature Store → Model API → Decision Engine → Case Management**. Lo que cambia respecto a un sistema antifraude operado por el cliente final es **quién opera cada capa**: en nuestro modelo, todas son nuestras.

---

## 1. Vista general del flujo

```
Transaction → Feature Store → Model API → Decision Engine → Case Management
   (PyME)                                                        ↓
                                                          Analista interno
                                                                 ↓
                                                       Reporte quincenal
                                                          (a la PyME)
                                                                 ↓
                                                    (feedback re-entreno)
```

| capa | qué hace | quién la opera | latencia objetivo |
|---|---|---|---:|
| Transaction | recibe la compra entrante del checkout PyME | cliente (su checkout) | — |
| Feature Store | enriquece con histórico y rolling stats | nosotros (backend) | < 50 ms |
| Model API | calcula score con el modelo XGBoost | nosotros (servicio) | < 100 ms |
| Decision Engine | traduce score en acción (auto-aprobar/escalar/auto-bloquear) | nosotros (servicio) | < 50 ms |
| Case Management | cola de revisión interna con SHAP traducido | nosotros (analista) | asíncrono |
| **Total online** | (Transaction → Decision Engine) | | **< 300 ms** |

> El presupuesto de 300 ms es estándar de industria para checkout no-friction. La PyME nunca percibe la latencia del antifraude — para su comprador, la decisión es instantánea. Solo los casos escalados al analista tardan más, pero esos no bloquean al comprador en tiempo real (se procesan asíncronamente).

---

## 2. Las 5 capas en detalle

### 2.1 Transaction

**Qué es**: el evento crudo que dispara todo el pipeline. Una compra entrante desde el checkout de la PyME (tienda Shopify/Tiendanube/WooCommerce).

**Quién lo opera**: el checkout de la PyME (no nosotros). Esta capa es donde nuestra **API se conecta** durante la integración asistida — la PyME no escribe código, nuestro equipo lo hace en 2-4 hrs.

**Qué contiene**:
- Datos del cargo: monto, moneda, timestamp.
- Identificador de tarjeta (hasheado / tokenizado, nunca el PAN completo).
- Metadata del merchant: ID, categoría MCC, ubicación.
- Contexto de la sesión: IP, user agent, dispositivo, geolocalización si está disponible.

**Qué NO contiene**:
- El histórico de la tarjeta (eso vive en nuestro Feature Store).
- La decisión del modelo (eso es el output, no input).
- Información personal identificable más allá de lo necesario para procesar el cargo (cumplimiento PCI-DSS).

> En el proyecto, esto es análogo al row crudo del CSV de Kaggle (`data/raw/fraudTrain.csv`). En producción real, viene de un webhook/API call del checkout de la PyME hacia nuestro servicio.

---

### 2.2 Feature Store

**Qué hace**: enriquece la transacción con el contexto histórico necesario para que el modelo pueda predecir. Sin esta capa, el modelo no tendría las features con las que fue entrenado.

**Quién lo opera**: nosotros. El cliente PyME no necesita saber que existe.

**Qué contiene** (las 27 features del proyecto, ver `methodology.md` § Fase 2):

| categoría | features | de dónde vienen |
|---|---|---|
| monto | `log1p_amt`, `amt_gt_p95_legit` | de la transacción + p95 precalculado |
| temporal | `hour`, `dow`, `is_night`, `hour_sin/cos` | derivadas del timestamp |
| demografía | `age` | del registro del titular (precalculado) |
| geografía | `time_since_last_tx`, `dist_consecutive_km`, `velocity_kmh` | requieren la transacción anterior de la cc |
| histórico (rolling) | 12 features de 1h/24h/7d | requieren el histórico de la cc |
| categórico | `te_merchant`, `te_category`, `te_state`, `te_job` | mapping precalculado en train |

**Por qué es crítica**:
- Calcular en tiempo de inferencia el rolling de 24h por tarjeta significaría escanear toda la base de datos en cada predicción. Inviable a 300 ms.
- El Feature Store mantiene esos agregados precalculados y actualizados continuamente. Cada nueva transacción suma incrementalmente al rolling de su tarjeta.
- Garantiza que las features en producción se calculan con las **mismas reglas** que en entrenamiento (`closed='left'`, target encoding fitteado solo en train, etc.). Si la regla cambia entre train e inference, el modelo se rompe silenciosamente.

> Esto es lo que en MLOps se llama "training-serving skew prevention". Es la causa más común de modelos que andan bien en evaluación y mal en producción.

---

### 2.3 Model API

**Qué hace**: recibe el feature vector enriquecido, devuelve un score de fraude entre 0 y 1.

**Quién lo opera**: nosotros (servicio HTTP interno).

**Qué carga**:
- `models/xgb_best.json` — el modelo XGBoost entrenado en Fase 3.
- `models/calibrator.pkl` — el calibrador isotónico (cuando se requiere score como probabilidad real, no solo ranking).

**Qué devuelve** (consumo interno — nuestro Decision Engine es quien recibe esto):
```json
{
  "score_raw": 0.872,
  "score_calibrated": 0.845,
  "shap_top_features": [
    {"feature": "log1p_amt", "value": 7.2, "shap": 0.35},
    {"feature": "rolling_amt_mean_24h", "value": 12.5, "shap": 0.28},
    {"feature": "amt_gt_p95_legit", "value": 1, "shap": 0.15}
  ],
  "model_version": "xgb_best_2025-04",
  "latency_ms": 87
}
```

**Por qué SHAP en línea**:
El score crudo (0.872) no le sirve a nuestro analista — necesita saber **por qué** para tomar la decisión. Calcular el SHAP en tiempo de predicción agrega latencia (típicamente +30-50 ms con TreeExplainer), pero a cambio el caso llega al Case Management con la explicación lista. Sin esto, el analista tiene que abrir herramientas separadas y rastrear las features manualmente, lo que multiplica el tiempo de resolución por 5-10×.

> Decisión de diseño: aceptamos +50 ms de latencia para que el Case Management reciba contexto explicable. Es un trade-off documentado. Esos mismos SHAP features (traducidos a lenguaje natural) son los que después aparecen en el reporte quincenal del cliente.

---

### 2.4 Decision Engine

**Qué hace**: traduce el score numérico en una acción de negocio. Es la capa que separa "el modelo dice X" de "qué hacemos al respecto".

**Quién lo opera**: nosotros. Esta capa **no devuelve nada al cliente para decidir** — decide automáticamente o escala internamente al analista nuestro.

**Lógica de decisión** (basada en los thresholds del proyecto):

| score calibrado | acción | quién resuelve |
|---|---|---|
| < 0.30 | **Auto-aprobar** | nadie — decisión automática |
| 0.30 – 0.85 | **Escalar al analista** | analista interno (asíncrono) |
| ≥ 0.85 | **Auto-bloquear** | nadie — decisión automática |

> **Nota importante sobre los thresholds**: el threshold operativo F1\* del proyecto es 0.6642 (ver `model_evaluation.md`). En producción, ese score caería dentro de "escalar al analista". El `0.85` para auto-bloqueo es deliberadamente más alto para minimizar falsos positivos en la decisión más costosa (bloquear a un cliente real). El `0.30` para auto-aprobar es bajo para minimizar carga sobre el analista — la mayoría de transacciones (>95%) se resuelven aquí.

**Qué más vive acá**:
- **Reglas de compliance**: por ejemplo, "bloquear toda transacción desde un país sancionado", "requerir 3DS si el monto supera $X". Estas reglas no dependen del modelo y se aplican independientemente.
- **Allowlists / denylists** de tarjetas conocidas (la analista las mantiene desde Case Management).
- **Rate limiting** por tarjeta o por IP para prevenir card testing.
- **Lógica de escalamiento de degradación**: si el sistema está degradado (Model API caída), aplicar política conservadora por default (escalar todo al analista).

**Output** hacia el checkout de la PyME (lo que ve su sistema):
```json
{
  "decision": "approve" | "block" | "pending_review",
  "reason_code": "auto_approve" | "auto_block" | "manual_review",
  "case_id": "case_4821"
}
```

**Importante**: el checkout de la PyME procesa la transacción según `decision`. Si es `pending_review`, el comprador recibe un mensaje de "tu compra está en validación, te confirmamos por email en menos de 30 min" — no se le bloquea ni se le aprueba en el momento. La analista resuelve el caso asíncronamente y dispara el cobro o la cancelación según lo que decida.

---

### 2.5 Case Management

**Qué hace**: gestiona el flujo de las transacciones que requieren intervención humana. Es la herramienta de trabajo interna de nuestro analista.

**Quién lo opera**: **nosotros — específicamente, nuestro analista de fraude interno**. Una analista atiende ~20-50 PyMEs según volumen.

**Importante**: el cliente PyME nunca entra a Case Management. Esta es nuestra herramienta de operación. La PyME ve los resultados después en el reporte quincenal.

**Qué contiene**:
- **Cola de revisión** ordenada por prioridad (función del score, monto y SLA del cliente).
- **Vista por caso**: la transacción + el score + los top features SHAP traducidos a lenguaje natural + histórico reciente de la tarjeta + contexto del comercio.
- **Acciones disponibles**: aprobar, bloquear, escalar a supervisor, marcar como falso positivo, agregar a denylist.
- **Persistencia obligatoria de la razón en lenguaje natural** — esto es lo que después aparece textual en el reporte quincenal del cliente (transparencia total).
- **Capacidad de contactar al titular** de la tarjeta antes de decidir (en casos de monto alto), vía email/SMS/WhatsApp.

**Vista típica que ve el analista**:
```
┌─────────────────────────────────────────────────────────────┐
│ Caso #4821 · Tienda Aurora · Score: 0.745 · $2,450 MXN     │
├─────────────────────────────────────────────────────────────┤
│ Tarjeta: ****1234  ·  Merchant: ELECTRONICA_XYZ            │
│ Hora: 03:42 (madrugada)  ·  Ubicación: CDMX                │
│                                                             │
│ Por qué el modelo escaló esta transacción:                 │
│ • monto 5× superior al promedio de 24h de esta tarjeta     │
│ • supera el p95 de transacciones legítimas                 │
│ • horario nocturno (alta tasa base de fraude)              │
│                                                             │
│ Histórico (últimas 5 tx): $180, $220, $95, $310, $2,450    │
│                                                             │
│ [ Aprobar ]  [ Bloquear ]  [ Contactar titular ]  [ FP ]   │
│                                                             │
│ Razón obligatoria (aparecerá en reporte al cliente):       │
│ ┌─────────────────────────────────────────────────────────┐│
│ │ ____________________________________________________     ││
│ └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

> El mockup `mockup_panel_pyme_revision_alerta` (renderizado en chat en el bloque de customer journey) es exactamente esta pantalla. **No es una vista del cliente** — es nuestra herramienta interna del analista.

**Por qué es crítico para la operación**:
- Sin un buen Case Management, los analistas pierden el 60-70% de su tiempo buscando contexto en otros sistemas. Con SHAP integrado y vista unificada, resuelven cada caso en 4-8 minutos en lugar de 15-20.
- La razón en lenguaje natural que escribe el analista es **el contenido principal del reporte quincenal** que recibe la PyME — la transparencia depende de esto.
- La decisión del analista es el ground truth que alimenta el próximo re-entreno trimestral del modelo.

---

## 3. El loop de retroalimentación

El sistema no es lineal — tiene dos feedback loops distintos:

```
Loop 1 (al cliente PyME):
Analista resuelve caso → Razón persistida → Cada 15 días → Reporte quincenal por email

Loop 2 (al modelo):
Analista resuelve caso → Decisión persistida como label → Almacenamiento histórico
                              ↓
                  Re-entreno mensual o trimestral del modelo
                              ↓
            Validación canary (5% del tráfico)
                              ↓
              Promoción al 100% si métricas OK
```

**Loop 1 — Reporte al cliente (quincenal)**:
- Cada 15 días enviamos un PDF + email a cada PyME activa.
- Contiene: cifras agregadas + detalle de cada caso que resolvió la analista, con la razón en lenguaje natural.
- Incluye recomendaciones de la analista basadas en tendencias detectadas.
- Es la única visibilidad que el cliente PyME tiene sobre la operación.

**Loop 2 — Re-entreno del modelo**:
- **Frecuencia**: mensual para ajustes de sensibilidad, trimestral para re-entrenos completos.
- **Validación canary** — la nueva versión se despliega al 5% del tráfico antes de promover al 100%.
- **Rollback automático** si las métricas del canary se degradan respecto a la versión anterior.
- **Por qué no continuo**: los chargebacks confirmados por el banco emisor llegan con 30-60 días de delay. Las decisiones del analista son provisionales hasta que se confirmen. El re-entreno continuo (online learning) tiene además riesgo de aprender ataques adversariales en tiempo real.

---

## 4. Restricciones operativas y SLAs

| restricción | valor objetivo | qué pasa si se rompe |
|---|---:|---|
| Latencia p99 end-to-end (capas online) | < 300 ms | el checkout PyME ve timeout, customer experience del comprador se degrada |
| Disponibilidad del Model API | 99.9% | fallback a reglas duras del Decision Engine (escalar todo al analista) |
| Throughput sostenible | 100 tx/seg por instancia | escalado horizontal automático |
| Tiempo desde escalamiento hasta resolución (analista) | < 30 min para horario de oficina, < 4 hrs noches/findes | comprador queda esperando confirmación de su pedido |
| Tiempo de resolución promedio del analista por caso | 4-8 min | cola crece, productividad cae |
| Falsos positivos por PyME por quincena | 0 | cliente legítimo bloqueado daña la marca de la PyME |
| Frecuencia de reporte quincenal | cada 15 días, sin excepciones | rompe la promesa de transparencia |
| Frecuencia de re-entreno modelo | trimestral mínimo | drift sin corrección, modelo se degrada |

---

## 5. Cómo encaja el modelo del proyecto en esta arquitectura

| capa de la arquitectura | qué del proyecto vive aquí |
|---|---|
| Transaction | análogo al row crudo del CSV de Kaggle |
| Feature Store | la lógica de `src/features.py` ejecutada incrementalmente |
| Model API | `models/xgb_best.json` cargado en un servicio HTTP |
| Decision Engine | thresholds derivados del análisis costo-beneficio (`model_evaluation.md` § 1) + reglas de negocio nuestras |
| Case Management | aún no implementado en el proyecto — es trabajo futuro de Fase 5 |
| Loop de retroalimentación | parcialmente cubierto por el pipeline de re-entreno (`make all`) |

**Lo que el proyecto SÍ tiene resuelto**:
- Modelo entrenado y validado (Fases 1-4 completas).
- Feature engineering anti-leakage que se puede portar 1:1 al Feature Store.
- Calibrador isotónico para el caso de uso de probabilidad.
- SHAP funcionando con propiedad de aditividad verificada.
- Análisis costo-beneficio que define los thresholds del Decision Engine.

**Lo que el proyecto NO tiene (y se reconoce explícitamente)**:
- API HTTP para servir el modelo en línea.
- Feature Store productivo (cálculo en streaming).
- Interfaz de Case Management para nuestro analista.
- Pipeline de re-entreno automatizado con validación canary.
- Métricas de observabilidad (latencia, throughput, drift).
- Sistema de generación automática del reporte quincenal.

> Estos componentes son ingeniería de plataforma + producto, no modelado. Son trabajo de Fase 5+ que está fuera del alcance del capstone pero documentado acá para mostrar que el camino a producción está pensado.

---

## 6. Quiénes interactúan con el sistema

A diferencia de un sistema antifraude operado por un banco grande (donde el cliente final tiene su propio equipo antifraude), en nuestro modelo **casi todos los actores operativos son nuestros**. La PyME es un cliente pasivo.

| actor | de qué lado | dónde toca el sistema | qué necesita |
|---|---|---|---|
| Comprador final | externo | Transaction (input) | que la decisión sea rápida; si su compra es legítima, que no se bloquee |
| Checkout de la PyME | externo (integrado vía API) | Transaction + recibe Decision | latencia < 300 ms, decisión clara (approve/block/pending) |
| Cliente PyME (dueño/operador) | externo (cliente nuestro) | recibe **reporte quincenal** | confianza en el servicio, transparencia, contacto humano (analista) |
| Cliente legítimo bloqueado | externo (afectado) | recibe email post-decisión | canal claro para resolver con la PyME |
| **Nuestro analista de fraude** | interno | Case Management | contexto rico (SHAP), herramientas de contacto al titular, registro de razón |
| **Nuestro equipo de soporte** | interno | onboarding + soporte continuo | tooling para ajustar sensibilidad, resolver dudas de la PyME |
| **Nuestro equipo de MLOps** | interno | toda la arquitectura | observabilidad, alertas de drift, capacidad de rollback |
| **Nuestro equipo de compliance** | interno | Model API + Case Management | trazabilidad de cada decisión, model card actualizada, auditorías |

> **El diferenciador frente a Stripe Radar**: ellos venden la infraestructura (Transaction → Decision Engine) y el cliente opera Case Management. Nosotros vendemos el servicio completo — Case Management también es nuestro. Para una PyME sin equipo antifraude, esa diferencia es la propuesta de valor.

---

## 7. Tres salvedades honestas

1. **No tenemos esta arquitectura implementada**, solo diseñada. El proyecto entrenó el modelo (Fases 1-4); la operación en producción es Fase 5+ y está documentada conceptualmente, no codificada.

2. **Las latencias objetivo son estándar de industria**, no medidas en este sistema específico. Sirven como guía de diseño pero requerirían validación en un piloto real.

3. **El modelo de servicio depende de la productividad del analista**. Si una analista atiende 20 PyMEs y cada una genera 3-5 casos/semana, son 60-100 casos/semana — manejable. Si la sensibilidad del modelo se calibra mal y genera 30 casos/semana por PyME, los costos del servicio explotan. La gestión de la cola del analista es un parámetro operativo crítico que no está cubierto en este diseño abstracto.

---

## 8. TL;DR

- El sistema procesa una transacción end-to-end en **< 300 ms** (capas online).
- Cinco capas: **Transaction → Feature Store → Model API → Decision Engine → Case Management**.
- **Todas las capas operativas son nuestras** — la PyME no opera nada. Esto es el modelo de servicio gestionado.
- El modelo del proyecto (XGBoost + SHAP + calibrador) vive en el Model API.
- Los thresholds del Decision Engine definen tres rutas: auto-aprobar (< 0.30), escalar a analista (0.30–0.85), auto-bloquear (≥ 0.85).
- **El Decision Engine devuelve `pending_review` al checkout** cuando escala — el comprador queda esperando confirmación, no se le bloquea ni aprueba en el momento.
- **Case Management es nuestra herramienta interna**, no del cliente. La PyME ve los resultados después en el reporte quincenal.
- Los **dos loops de retroalimentación**: Loop 1 al cliente (reporte quincenal), Loop 2 al modelo (re-entreno trimestral).
- Lo que el proyecto **no implementó pero diseñó**: API HTTP, Case Management, pipeline de re-entreno automatizado, generación de reportes.

> Para los slides 3-5 del deck final: el diagrama horizontal de las 5 capas + los 3 puntos del TL;DR cubren el feedback del profesor sobre arquitectura E2E real-time. El detalle de cada capa queda como apéndice de Q&A.

---

## Documentos relacionados

- [`methodology.md`](methodology.md) — pipeline de entrenamiento (qué genera el `xgb_best.json` que vive en el Model API).
- [`model_evaluation.md`](model_evaluation.md) — análisis costo-beneficio que justifica los thresholds del Decision Engine.
- [`architecture_decisions.md`](architecture_decisions.md) — ADRs del proyecto (decisiones técnicas del modelo, no del sistema productivo).
- [`kpis_financieros.md`](kpis_financieros.md) — análisis de mercado y benchmarks de la industria para el segmento PyME.
- [`customer_journey.md`](customer_journey.md) — cómo viven el sistema los actores externos (cliente PyME, comprador final).
- `mockup_panel_pyme_revision_alerta` — mockup del Case Management (vista del analista interno).
- `mockup_reporte_quincenal_pyme` — mockup del reporte que recibe el cliente PyME (Loop 1 del feedback).

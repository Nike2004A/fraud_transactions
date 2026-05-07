# Governance y explainability — cómo defendemos el sistema

> **Veredicto corto**: nuestro servicio antifraude toma decisiones automáticas que afectan ventas reales de PyMEs y experiencias de compradores reales. Como cualquier sistema de IA aplicado a decisiones financieras, requiere un marco de governance que cubra cuatro dimensiones: (1) cómo protegemos los datos personales que procesamos, (2) cómo explicamos las decisiones del modelo, (3) cómo verificamos que el modelo no discrimina sistemáticamente a ciertos grupos, y (4) cómo dejamos rastro auditable de cada decisión. Este documento describe el framework operativo, no aspiracional.
>
> Documento orientado a responder el feedback del profesor sobre governance y explainability. El alcance es deliberadamente acotado: cubre lo mínimo defendible para una operación seria, sin entrar en compliance regulatorio profundo (que requeriría un equipo legal especializado).

---

## 1. Por qué importa esta sección

Tres razones concretas:

1. **Decisiones automáticas con consecuencias reales**: cuando el modelo bloquea una transacción, un comprador legítimo puede quedar afectado. Cuando el modelo deja pasar un fraude, una PyME pierde dinero. Ninguna de las dos situaciones es trivial — necesitamos poder explicar por qué pasó.

2. **El servicio toca datos sensibles**: número de tarjeta hasheado, monto, ubicación, comportamiento de compra. La Ley Federal de Protección de Datos Personales (LFPDPPP) en México regula esto, y la reforma de marzo 2025 la endureció.

3. **El cliente PyME confía sin operar**: a diferencia de un producto self-service (donde el cliente configura sus propias reglas y asume responsabilidad), nuestro modelo es servicio gestionado. La PyME nos delega completamente la decisión. Esa delegación obliga a un nivel de transparencia y trazabilidad mayor.

> Si una PyME nos pregunta "¿por qué bloquearon a mi mejor cliente?", o si un regulador llega con una auditoría, o si un comprador denuncia discriminación — necesitamos respuestas concretas, no aspiraciones.

---

## 2. Pilar 1 — Protección de datos

### Qué datos procesamos

| categoría | dato | sensibilidad |
|---|---|---|
| Identificadores de tarjeta | hash del PAN, BIN (primeros 6 dígitos), últimos 4 | media — nunca el PAN completo |
| Monto y moneda | de cada transacción | baja |
| Metadata temporal | timestamp, hora, día | baja |
| Geografía | IP, ciudad, dirección de envío | media |
| Identificadores del comprador | email hasheado, teléfono hasheado | media |
| Histórico agregado | rolling stats por tarjeta (sin PII) | baja |
| Datos del comercio | merchant ID, categoría MCC, vertical | baja |

### Lo que NO procesamos

- **El PAN completo nunca toca nuestros sistemas** — el procesador (Conekta, Mercado Pago, Stripe) lo tokeniza antes de enviarnos la transacción.
- **No procesamos nombre real del titular**, solo el hash del email.
- **No procesamos información biométrica**, datos de salud, religiosos ni políticos.

### Cómo cumplimos LFPDPPP

| obligación legal | cómo lo hacemos |
|---|---|
| Aviso de privacidad explícito | la PyME firma contrato que describe qué datos procesamos en su nombre |
| Consentimiento del titular | la PyME es responsable del consentimiento del comprador (lo recolecta en su checkout) |
| Finalidad limitada | datos solo se usan para detección de fraude — no se comparten ni venden a terceros |
| Retención limitada | datos transaccionales: 24 meses (necesario para re-entreno). PII tokenizada: 5 años (auditoría) |
| Derecho ARCO del titular | el comprador puede pedir acceso/rectificación/cancelación a través de su PyME, que escala a nosotros |
| Encripción en reposo y tránsito | AES-256 en reposo, TLS 1.3 en tránsito |
| Aislamiento por cliente | cada PyME tiene un namespace lógico — un analista nunca ve datos de otra PyME en el mismo caso |

> **Nota importante**: PCI-DSS (estándar de industria de tarjetas) recae principalmente sobre el procesador, no sobre nosotros, porque nunca tocamos el PAN completo. Esto es deliberado en el diseño: minimiza nuestra exposición regulatoria.

---

## 3. Pilar 2 — Transparencia del modelo

### Tres niveles de explicabilidad

El sistema explica sus decisiones a tres audiencias distintas, en tres niveles de profundidad:

| audiencia | qué necesita | qué le damos |
|---|---|---|
| Cliente PyME | confianza en cada decisión escalada | razón en lenguaje natural en el reporte quincenal |
| Nuestro analista | contexto técnico para decidir | top features SHAP + traducción + histórico |
| Auditor / regulador | trazabilidad completa | model card + log inmutable + reproducibilidad |

### Model card del sistema

Mantenemos un documento vivo que describe el modelo en producción. Las secciones obligatorias:

| sección | contenido |
|---|---|
| Descripción del modelo | XGBoost binario, calibrador isotónico, 27 features anti-leakage |
| Caso de uso previsto | detección de fraude en transacciones de tarjeta de crédito en e-commerce |
| Caso de uso fuera de alcance | scoring crediticio, KYC, antimoney laundering — no usamos este modelo para esos fines |
| Métricas de desempeño actual | PR-AUC, ROC-AUC, recall@1%, precisión y recall al threshold operativo, métricas por segmento |
| Datos de entrenamiento | descripción del dataset (sin PII), tamaño, ventana temporal, distribución de clases |
| Limitaciones conocidas | dataset semi-sintético, top features dominadas por monto (vulnerable a smurfing), gap day vs night |
| Frecuencia de re-entreno | trimestral mínimo, mensual si métricas se degradan |
| Versión del modelo en producción | identificador único, fecha de despliegue, métricas de validación canary |
| Responsable | analista de fraude líder + responsable de MLOps |
| Última auditoría | fecha de revisión interna del modelo |

> La model card del modelo del proyecto (XGBoost entrenado en Fase 3) está en [`model_evaluation.md`](model_evaluation.md). En producción, se versiona y actualiza cada re-entreno.

### SHAP en cada decisión

Cada predicción del modelo viene acompañada de los top features SHAP (ver [`arquitectura_realtime.md`](arquitectura_realtime.md) § Model API). Estos features se traducen a lenguaje natural en dos puntos:

1. **Internamente**, en el Case Management que ve nuestro analista (técnico, con valores numéricos).
2. **Externamente**, en el reporte quincenal que recibe la PyME (lenguaje natural, sin jerga).

La tabla de traducción está documentada en [`customer_journey.md`](customer_journey.md) § 3.

> El test de aditividad SHAP está verificado: `sum(SHAP) + base_value ≈ logit(score)` con tolerancia 1e-4. Documentado en `tests/test_evaluate.py`.

---

## 4. Pilar 3 — Fairness

Esta sección es la más sensible y la que más probablemente reciba preguntas en el panel. La cubrimos con seriedad.

### Qué entendemos por fairness en este contexto

**Fairness no significa "el modelo trata a todos igual"** — el modelo trata a todos según el riesgo estimado, lo cual es deseable. Fairness significa que el modelo **no comete errores sistemáticamente más frecuentes en ciertos grupos protegidos** sin justificación.

En detección de fraude, el error costoso para el cliente final es el **falso positivo** (bloquear a un cliente legítimo). Si el modelo bloquea por error mucho más a compradores de cierto estado, edad o género, eso es discriminación algorítmica — aunque el modelo no use directamente esos atributos como features.

### Qué medimos

| métrica | qué mide | umbral de alerta |
|---|---|---|
| **FPR por estado** (32 estados de México) | tasa de falsos positivos por entidad federativa | desviación >2x respecto a la mediana |
| **FPR por grupo etario** (<30, 30-50, 50+) | tasa de falsos positivos por edad del titular | desviación >1.5x |
| **FPR por categoría de comercio** (top-10 verticales) | tasa de falsos positivos por vertical | desviación >2x |
| **Recall por estado** | sensibilidad del modelo por entidad | desviación >2x respecto a la mediana |
| **Volumen de transacciones por grupo** | cobertura mínima para considerar la métrica válida | mínimo 100 transacciones/grupo/trimestre |

> No medimos por género porque no procesamos el género del titular. No medimos por raza/etnia porque esos datos no están disponibles ni son apropiados de inferir. Lo que medimos por estado/edad/vertical es lo que tenemos disponible y lo que tiene sentido para el caso de uso.

### Por qué medimos esto y no otra cosa

El modelo del proyecto tiene una limitación honesta: una de las features (`te_state`) es target encoding por estado. Si históricamente ciertos estados tienen más fraude reportado, el modelo aprende ese patrón. Eso es **estadísticamente correcto** pero **éticamente delicado** — puede crear un loop donde compradores de ciertos estados son sistemáticamente revisados con más sospecha.

Por eso medir FPR por estado es nuestra alerta más importante. Si un estado tiene FPR significativamente más alto, no significa necesariamente que el modelo esté roto — pero **obliga a investigar y, si corresponde, intervenir**.

### Qué hacemos cuando detectamos un sesgo

1. **Investigación causal**: ¿el sesgo viene de los datos de entrenamiento o del modelo? Generalmente viene de los datos (chargebacks históricos están sesgados por dónde los bancos mexicanos detectan más fácil).
2. **Re-balanceo del dataset**: ajuste de pesos por grupo subrepresentado en el siguiente re-entreno.
3. **Threshold por segmento**: en casos extremos, ajustar el threshold del Decision Engine para grupos donde el modelo es menos confiable.
4. **Intervención humana adicional**: bajar el umbral de escalamiento al analista para grupos donde el modelo tiene sesgo conocido.
5. **Documentación obligatoria**: cada intervención queda registrada en la model card y se reporta en la auditoría trimestral.

### Frecuencia de revisión

| revisión | frecuencia | quién |
|---|---|---|
| Métricas de fairness automáticas | mensual | sistema de monitoreo |
| Revisión por analista de fraude líder | trimestral | analista líder |
| Auditoría de fairness completa | anual | revisor externo (idealmente académico o consultor independiente) |

### Limitaciones honestas de nuestro fairness check

1. **No medimos lo que no podemos medir**: género, raza, religión, orientación sexual, condición socioeconómica — no tenemos esos datos y no los queremos tener.
2. **El sesgo del dataset es una herencia**: si el dataset histórico tenía sesgos sistémicos, los heredamos. La medición sirve para detectar y corregir, no para garantizar ausencia de sesgo.
3. **El sesgo geográfico es estructural**: ciertos estados sí tienen más fraude reportado. Diferenciar entre patrón real y sesgo histórico es difícil sin contrafactual.

---

## 5. Pilar 4 — Auditabilidad

### Qué se loggea de cada transacción

| dato | retención |
|---|---|
| Input completo enviado al modelo (features) | 24 meses |
| Versión del modelo que tomó la decisión | indefinido (asociado al log) |
| Score crudo + score calibrado | 24 meses |
| SHAP top features | 24 meses |
| Decisión final (auto-aprobar / escalar / auto-bloquear) | 5 años |
| Si fue escalada al analista: decisión humana + razón en lenguaje natural | 5 años |
| Identificador del analista que decidió | 5 años |
| Reglas de negocio aplicadas (allowlist, denylist, compliance) | 5 años |

### Por qué retención de 5 años para decisiones

- Plazo de prescripción de obligaciones civiles en México (varía 2-10 años según el caso).
- Posibles auditorías regulatorias futuras (CNBV, CONDUSEF) si una PyME cliente crece y se vuelve regulada.
- Defensa ante reclamos de discriminación o trato indebido.

### Inmutabilidad

El log de decisiones se escribe en un sistema **append-only**: una vez registrada una decisión, no se puede modificar ni borrar dentro del periodo de retención. Si se descubre un error, se agrega un nuevo registro de corrección, pero el original queda.

### Reproducibilidad

Para cualquier decisión histórica, debemos poder responder:

- **¿Qué versión del modelo decidió?** → del log.
- **¿Cuáles fueron los inputs exactos?** → del log.
- **¿Por qué se decidió así?** → SHAP del log + razón del analista si aplica.
- **¿Qué hubiera decidido el modelo actual con los mismos inputs?** → re-ejecución sobre el modelo vigente.

> Esto no es trivial: requiere versionar el modelo, el código de feature engineering, y los datos de referencia (mappings de target encoding, etc.). El proyecto del capstone ya tiene esta infraestructura parcialmente — MLflow versiona los runs y `models/xgb_best.json` se persiste por versión.

---

## 6. Cómo se conecta con el resto del sistema

| capa de la arquitectura | gobernanza relevante |
|---|---|
| Transaction (input) | encripción TLS 1.3, validación de schema, no se persiste PAN completo |
| Feature Store | features documentadas en model card, anti-leakage verificado por tests |
| Model API | versión del modelo loggeada, SHAP por decisión, latencia monitoreada |
| Decision Engine | reglas de negocio documentadas, thresholds versionados, decisiones loggeadas |
| Case Management | razón obligatoria en lenguaje natural, identificador del analista |
| Reporte quincenal | transparencia hacia el cliente PyME, alineado con razón del analista |

> El framework de governance no es una capa separada — está embebido en cada componente del sistema. Ver [`arquitectura_realtime.md`](arquitectura_realtime.md).

---

## 7. Marco regulatorio mexicano de referencia

Aclaración importante: **no somos una entidad regulada por CNBV**. Somos un proveedor SaaS B2B de software antifraude. La regulación financiera mexicana aplica a nuestros clientes (PyMEs y procesadores) más que a nosotros directamente. Sin embargo, hay tres marcos que sí nos tocan:

### LFPDPPP — Ley Federal de Protección de Datos Personales en Posesión de Particulares

**Qué exige**: aviso de privacidad, consentimiento, finalidad limitada, derecho ARCO, medidas de seguridad, notificación de brechas. Reformada en marzo 2025 con requisitos más estrictos sobre datos biométricos (no nos aplica directamente porque no procesamos biometría) y notificación de incidentes.

**Cómo cumplimos**: contrato con cada PyME que documenta el tratamiento, implementación técnica de los pilares 1 y 4 de este documento.

### CNBV — Reglas sobre outsourcing financiero (2024)

**Qué exige**: si una entidad regulada (banco, IFPE, IFC, SOFOM, SOFIPRO) contrata servicios externos relacionados con su operación financiera, el outsourcing debe cumplir requisitos de trazabilidad, auditoría, controles documentados. La entidad regulada sigue siendo responsable ante CNBV.

**Cómo nos toca**: si nuestros clientes incluyen una IFPE o SOFOM (poco probable para PyMEs e-commerce, pero posible si crecemos a clientes mayores), necesitan que cumplamos requisitos de auditoría y trazabilidad. El pilar 4 de este documento está diseñado pensando en esto.

### Ley Fintech 2.0 (2025) — clasificación de riesgo de IA

**Qué exige**: la reforma 2025 introdujo un modelo de clasificación de riesgo para sistemas de IA inspirado en el AI Act europeo, con foco en lending, credit scoring y fraud detection. Aunque está dirigido a entidades reguladas, define expectativas de la industria.

**Cómo nos toca**: nuestro sistema cae en una categoría de "riesgo medio" según los criterios europeos análogos (decisión automática con impacto financiero, pero sin riesgo de vida o derechos fundamentales). Las prácticas que aplica el pilar 2 (transparencia, model card, explainability) están alineadas con estas expectativas.

### Lo que NO está regulado y por qué importa

**No hay regulación mexicana específica que obligue a un proveedor SaaS antifraude no regulado a hacer fairness audits**. Lo hacemos voluntariamente porque:

- Es la dirección regulatoria global (EU AI Act, propuestas en US y LATAM).
- Es buena práctica de la industria.
- Reduce nuestro riesgo legal en caso de denuncias de discriminación.
- Es defendible frente a clientes y al panel.

---

## 8. Las salvedades honestas

1. **Este framework está diseñado, no implementado en su totalidad**. El proyecto del capstone tiene model card (`model_evaluation.md`), tests anti-leakage, y SHAP funcional. No tiene todavía el log inmutable de 5 años, las métricas de fairness automatizadas, ni el sistema de re-ejecución para auditoría histórica. Son trabajo de Fase 5+.

2. **No tenemos abogado de planta**. Las menciones a LFPDPPP, CNBV y Ley Fintech 2.0 reflejan investigación responsable pero no son asesoría legal. Antes de operar comercialmente, este framework requiere validación por un abogado especializado en fintech mexicano.

3. **El fairness check es una intención, no una garantía**. Medir FPR por estado/edad/vertical detecta sesgos visibles, pero no garantiza ausencia de discriminación sutil. La ausencia de evidencia no es evidencia de ausencia.

4. **El analista humano introduce su propio sesgo**. Cuando el modelo escala un caso, la decisión final la toma una persona — con sus propios sesgos cognitivos. Mitigamos esto con razón obligatoria documentada y revisión cruzada periódica, pero no lo eliminamos.

5. **El servicio gestionado concentra responsabilidad en nosotros**. A diferencia de un producto self-service donde el cliente toma sus propias decisiones, en nuestro modelo cada error operativo es nuestro. Eso es bueno para el cliente pero exige el rigor de governance que este documento describe.

---

## 9. TL;DR

- **Cuatro pilares**: protección de datos, transparencia del modelo, fairness, auditabilidad.
- **No somos entidad regulada**, pero la LFPDPPP, las reglas CNBV de outsourcing y la Ley Fintech 2.0 definen el marco que aplicamos voluntariamente.
- **Tres niveles de explicabilidad**: razón en lenguaje natural para la PyME, SHAP traducido para el analista, model card completa para auditoría.
- **Fairness se mide concretamente**: FPR y recall por estado, edad, vertical. Trimestralmente, con umbrales de alerta y plan de intervención si se dispara.
- **Log inmutable de 5 años** para cada decisión, con reproducibilidad garantizada.
- **Cinco salvedades honestas**: framework diseñado más que implementado, no es asesoría legal, fairness es intención no garantía, el analista humano tiene su propio sesgo, el servicio gestionado concentra responsabilidad.

> Para los slides 12-13 del deck final: el diagrama de los 4 pilares + el TL;DR cubren el feedback del profesor sobre governance y explainability. Las preguntas técnicas específicas (cómo se mide fairness, qué exige LFPDPPP) tienen respuesta detallada en este documento como apéndice de Q&A.

---

## Documentos relacionados

- [`arquitectura_realtime.md`](arquitectura_realtime.md) — la arquitectura donde se embeben los controles de governance.
- [`model_evaluation.md`](model_evaluation.md) — model card actual del modelo del proyecto (Pilar 2).
- [`customer_journey.md`](customer_journey.md) — cómo se manifiesta la transparencia hacia el cliente PyME (reporte quincenal).
- [`architecture_decisions.md`](architecture_decisions.md) — ADRs que documentan decisiones técnicas auditables (Pilar 4).
- `framework_governance_explainability` — diagrama visual de los 4 pilares (renderizado en chat).

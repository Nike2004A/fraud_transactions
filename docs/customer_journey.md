# Customer Journey — el cliente PyME e-commerce

> **Veredicto corto**: el cliente del producto es una PyME mexicana que vende en Shopify, Tiendanube o WooCommerce. Nuestro modelo de entrega es un **servicio gestionado de antifraude**: nosotros integramos la API en su checkout (2-4 horas), el modelo decide automáticamente la mayoría de los casos, un analista de fraude de nuestro equipo revisa los casos ambiguos, y la PyME recibe un reporte quincenal por email con el detalle completo. La PyME no opera ningún sistema, no instala plugins, no decide casos. Solo recibe los reportes y puede contactar a su analista por WhatsApp si tiene dudas.
>
> Documento orientado a responder el feedback del profesor sobre customer journey. A diferencia de un sistema antifraude operado por un banco grande (con departamento antifraude propio) o un producto self-service tipo Stripe Radar (donde el comerciante configura todo), nosotros operamos el servicio completo en nombre de la PyME.

---

## 1. Quién es nuestro cliente

| dimensión | perfil |
|---|---|
| Tipo de negocio | PyME e-commerce mexicana |
| Plataforma | Shopify / Tiendanube / WooCommerce |
| Volumen típico | 200-2,000 transacciones/mes |
| Equipo | 1-5 personas en total |
| Quién opera el sistema antifraude | **nadie del lado del cliente** — lo operamos nosotros |
| Conocimiento técnico | irrelevante — no necesita usar nada técnico |
| Lo que sí entiende | "te ahorramos $8,420 este mes", "tu analista decidió esto por estas razones", "WhatsApp si tenés dudas" |

> **Decisión de diseño clave**: el producto es un servicio gestionado, no una herramienta. El cliente PyME no toma decisiones operativas — paga para que alguien (nosotros) se encargue del problema entero. Esto cambia radicalmente el journey vs un producto self-service.

---

## 2. El modelo de servicio en una imagen

| capa | quién la opera | qué hace |
|---|---|---|
| **API en checkout** | nuestro equipo (integración inicial) | recibe cada transacción, devuelve decisión |
| **Decisiones automáticas** (alta y baja confianza) | el modelo XGBoost | aprueba o bloquea sin intervención humana |
| **Casos ambiguos** (zona gris) | **analista de fraude interno** | revisa con SHAP + contexto, decide |
| **Dashboard operativo** | nuestro equipo | herramienta interna, la PyME no entra |
| **Reporte quincenal** | nuestro equipo | PDF + email cada 15 días con detalle |
| **Soporte directo** | la analista asignada | WhatsApp/email para dudas del cliente |

La PyME solo interactúa con dos cosas: el **email con el reporte** cada 15 días, y el **WhatsApp de su analista** si tiene dudas. Nada más.

---

## 3. Los 5 momentos del journey

| # | momento | qué siente la PyME | qué pasa de nuestro lado |
|---|---|---|---|
| 1 | **Descubre el problema** | dolor real, varios contracargos en el mes | aparecemos en su búsqueda |
| 2 | **Integración asistida** | alivio + curiosidad | nosotros conectamos la API en 2-4 hrs |
| 3 | **Primer reporte** | curiosidad + confianza | el analista resolvió los primeros casos ambiguos |
| 4 | **Operación pasiva** | tranquilidad | servicio funcionando solo en background |
| 5 | **Renueva** | satisfacción | decisión obvia, switching cost alto |

### Momento 1 — Descubre el problema

**Trigger**: la PyME recibe el 3er o 4to contracargo del mes. Probablemente perdió entre $2,000 y $10,000 MXN en el mes (mercadería + chargeback fee).

**Comportamiento típico**:
- Se queja con su procesador (Conekta, Mercado Pago, Stripe).
- El procesador le dice "configurá tus reglas de fraude" — pero no sabe cómo.
- Busca en Google: "cómo evitar contracargos México", "antifraude Shopify México", "qué es 3D Secure".
- Encuentra nuestra propuesta en un blog de la AMVO o por recomendación de otro comerciante.

**Lo que necesita ver para convertir**:
- Mensaje claro: "Nosotros nos encargamos. Tú vendés tranquilo".
- Cifra concreta del problema: "Las PyMEs como vos pierden en promedio $4.61 por cada peso de fraude" (LexisNexis, dato verificable).
- Diferenciador: "No es una app que tenés que aprender a usar. Es un equipo que protege tu tienda".
- Un caso real: testimonio de otra tienda parecida con el ahorro mensual.
- **NO necesita ver**: PR-AUC, recall, SHAP, redes neuronales.

**Acción**: pide demo o llena un formulario corto en nuestro sitio. Lo contactamos en menos de 24 horas.

### Momento 2 — Integración asistida

**Acción del cliente**: cero código. Solo brinda credenciales y aprueba.

**Lo que hacemos nosotros**:
1. **Llamada inicial (30 min)**: entendemos su stack (qué plataforma, qué procesador, qué volumen, qué tipos de fraude está sufriendo).
2. **Integración técnica (2-4 hrs)**: nuestro equipo conecta nuestra API a su checkout vía:
   - Webhook nativo de la plataforma (caso Shopify y WooCommerce — fácil).
   - Integración custom con su procesador de pagos (Conekta/MercadoPago — requiere coordinación).
3. **Configuración inicial**: definimos la sensibilidad inicial (Conservador/Equilibrado/Agresivo) y los umbrales de monto para el escalamiento al analista. Defaults razonables, ajustables después.
4. **Presentación del analista asignado**: le presentamos a Mariana (o quien le corresponda) por WhatsApp/email. La PyME tiene contacto directo desde el día 0.

**Métrica clave del momento**: el cliente está protegido en menos de 5 días desde la primera llamada.

**Lo que NO hacemos**:
- Pedirle que escriba código.
- Darle un dashboard que tiene que aprender a usar.
- Pedirle configuración de reglas custom — eso lo hacemos nosotros basándonos en su contexto.

### Momento 3 — Primer reporte

**Trigger**: día 15 desde la activación, llega el primer reporte por email.

**Lo que recibe** (ver mockup `mockup_reporte_quincenal_pyme`):

> **Header**: nombre de tienda, periodo (ej. "16 al 30 de abril"), número de reporte.
>
> **Resumen** (3 cifras principales):
> - **Te ahorramos**: $8,420 MXN
> - **Pedidos revisados**: 312 (procesados sin esfuerzo)
> - **Falsos positivos**: 0 (ningún cliente real bloqueado)
>
> **Breakdown agregado**: de 312 pedidos, 305 aprobados automáticamente, 4 bloqueados automáticamente, 3 escalados a la analista.
>
> **Detalle de los 3 casos del analista** — para cada caso:
> - Pedido + monto + fecha + producto.
> - Estado final (Aprobado / Bloqueado).
> - Las 3 señales que vio el analista (en lenguaje natural — traducción de las features SHAP).
> - **Decisión y razón en prosa**: "bloqueado porque [...]. Mariana intentó contactar al titular [...]".
>
> **Recomendaciones del analista** al final: tendencias detectadas en la quincena, sugerencias accionables.
>
> **Footer**: WhatsApp y email de la analista para dudas.

**Por qué esto funciona**:
- La PyME entiende **qué pasó** (cifras agregadas) y **por qué** (detalle por caso).
- Cero jerga técnica. Cada caso explica el razonamiento humano, no el output del modelo.
- La transparencia genera confianza: no es una caja negra. Si la PyME tiene dudas sobre alguna decisión, puede preguntar.
- La analista tiene nombre y cara — humaniza el servicio.

### Momento 4 — Operación pasiva

**Frecuencia para la PyME**: cero acciones requeridas en el día a día. Recibe un reporte cada 15 días. Si tiene dudas, escribe por WhatsApp.

**Frecuencia desde nuestro lado**:
- **Diario**: el modelo procesa todas las transacciones automáticamente. La analista revisa los casos ambiguos en su cola interna (en promedio 2-5 por semana para una PyME mediana).
- **Cada caso del analista**: se documenta en nuestro sistema con la decisión y razón, listo para incluir en el próximo reporte.
- **Mensualmente**: se ajusta la sensibilidad si los falsos positivos suben o si el modelo está dejando pasar fraude.
- **Trimestralmente**: re-entrenamiento del modelo con los chargebacks confirmados (delay 30-60 días).

**Lo que la PyME percibe**: silencio operativo. Ningún cliente bloqueado por error, ninguna llamada del banco por contracargos, ningún drama. Solo el reporte que llega cada 15 días.

> Ese silencio es el producto. Es exactamente lo que la PyME quería desde el momento 1: olvidarse del problema.

### Momento 5 — Renueva

**Trigger**: termina el período del contrato (típicamente mensual o anual).

**Lo que vio antes de renovar**:
- 8 reportes quincenales (4 meses) o más, todos mostrando ahorro positivo.
- Cero contracargos en su procesador desde que activamos el servicio.
- Relación personal con su analista.

**Por qué renueva sin pensarlo**:
1. **ROI demostrado**: cada reporte muestra cuánto le ahorramos. Si ese número >> nuestro precio, decisión obvia.
2. **Cero fricción para irse**: no necesita migrar sistemas internos (no usaba ninguno), pero tampoco quiere romper la integración que ya funciona.
3. **Switching cost alto en lo positivo**: la API ya está en su checkout y funciona. Cambiar a otro proveedor implicaría re-integrar y volver a operar a riesgo durante semanas.
4. **Confianza personal**: ya conoce a su analista, sabe cómo trabaja, le contesta el WhatsApp.

**Métrica clave del momento**: tasa de renovación > 80%. Si es menor, el servicio no demostró suficiente valor.

---

## 4. Cómo trabajan los actores del servicio

### El analista de fraude (nuestro equipo)

**Quién es**: persona contratada por nosotros, dedicada a revisar los casos ambiguos de varios clientes PyME. Cada analista atiende ~20-50 PyMEs según volumen.

**Qué herramientas usa** (interno, la PyME no las ve):
- Cola de revisión priorizada por score y monto.
- Vista por caso con las 3 razones del modelo (SHAP traducido) + histórico de la tarjeta + contexto del comercio.
- Capacidad de contactar al titular de la tarjeta antes de decidir (en casos de monto alto).
- Persistencia de la decisión + razón en lenguaje natural (eso alimenta el reporte de la PyME).

**Qué decisiones toma**:
- Aprobar el pedido (si el caso ambiguo era falso positivo).
- Bloquear el pedido (si el caso ambiguo era fraude real).
- Escalar a supervisor (casos de monto muy alto o patrones nuevos).
- Marcar para revisión post-hoc cuando llegue el chargeback (etiqueta para re-entreno).

**Tiempo promedio por caso**: 4-8 minutos. La cola se mantiene por debajo de las 10 horas de espera.

> El mockup interno del analista es el `mockup_panel_pyme_revision_alerta` que ya generamos antes — sigue siendo válido, pero su usuario es nuestro analista, no el dueño de la PyME.

### El comprador legítimo (cliente final de la PyME)

**Cuándo aparece**: cuando bloqueamos por error una compra legítima (idealmente cero, en la práctica algunos casos).

**Qué necesita**:
- Saber inmediatamente que su compra fue rechazada.
- Un canal claro para resolverlo (WhatsApp del merchant).
- Reanudar la compra sin tener que empezar de cero.

**Lo que el sistema provee**:
- Email automático al cliente: "Tu compra requiere verificación, contacta a [merchant]".
- En caso de aprobación posterior por la analista: reanudación automática del pago si el cliente confirma.

> Este es nuestro mayor riesgo operativo: un falso positivo en un cliente recurrente daña la marca de la PyME. La analista mitiga este riesgo intentando contacto antes de bloquear casos de monto alto.

### Soporte (nuestro equipo)

**Cuándo interviene**:
- Onboarding técnico (primeras 2 semanas).
- La PyME tiene una duda que la analista no puede responder (típicamente facturación, técnica, cambios de plan).
- Incidente: caída del servicio, drift detectado, falso positivo recurrente que requiere ajuste de sensibilidad.

---

## 5. Lo que distingue este modelo

Comparación con otros enfoques posibles para defender el modelo frente al panel:

| dimensión | banco grande propio | self-service (Stripe Radar) | nuestro modelo |
|---|---|---|---|
| Quién opera el antifraude | departamento interno del banco | el comerciante | **nosotros** |
| Esfuerzo del cliente | alto (contratar, capacitar, retener) | medio (configurar reglas, ajustar) | **mínimo** (recibe reportes) |
| Conocimiento técnico requerido | alto | medio-alto | **cero** |
| Decisión de casos ambiguos | analista interno | el comerciante | **nuestro analista** |
| Visibilidad para el cliente | dashboard 24/7 | dashboard self-service | **reporte quincenal** |
| Ciclo de venta | N/A (es interno) | autoservicio inmediato | demo + setup en 5 días |
| Lock-in | alto (sistema propio) | bajo (cambia config) | **medio-alto** (API integrada) |

> Nuestro modelo no compite con Stripe Radar feature-por-feature — es un servicio diferente para un cliente diferente. Stripe vende infraestructura para que otros operen; nosotros operamos por el cliente.

### Por qué este modelo funciona para PyMEs

1. **Resuelve el problema real**: la PyME no quiere antifraude, quiere dejar de perder dinero por contracargos. Nuestro modelo lo logra sin que tenga que aprender nada.
2. **Match con su capacidad operativa**: una PyME de 3-5 personas no tiene tiempo para revisar alertas. Nosotros sí.
3. **Pricing alineado**: el costo del servicio (contrato mensual fijo) es predecible y se compara directamente con el ahorro del reporte.
4. **Diferenciación defendible**: el analista humano es difícil de comoditizar. Es nuestro foso competitivo más fuerte.

---

## 6. Las salvedades honestas

1. **Este modelo está diseñado, no validado**. No tenemos clientes reales todavía. El SLA de 4-8 minutos por caso, la conversión trial → renovación y la tasa de falsos positivos son hipótesis basadas en benchmarks de operación de fraude en LATAM, no mediciones nuestras.

2. **El modelo de negocio depende de la productividad del analista**. Si una analista atiende 20 PyMEs y cada PyME genera 3-5 casos por semana, son 60-100 casos por semana — manejable. Si genera 30 casos por semana cada PyME, los costos del servicio explotan. La sensibilidad del modelo se calibra para mantener el volumen de escalamientos sostenible.

3. **El operador de la integración inicial es un costo de adquisición**. 2-4 horas de trabajo de un ingeniero por cliente significa que el contrato mínimo viable tiene que justificarlo. Esto excluye PyMEs muy pequeñas (menos de 100 tx/mes) — quedan fuera del target.

4. **El switching cost alto del cliente es a doble filo**. Si nuestro servicio se degrada, el cliente sigue ahí porque irse cuesta. Eso es bueno para la retención pero nos exige mantener la calidad — un cliente atrapado y descontento es peor que un cliente que se va.

5. **WhatsApp como canal directo con el analista es supuesto cultural mexicano**. Funciona en México donde el 90%+ de las PyMEs operan por WhatsApp Business. En otros mercados (US, Europa) habría que ajustar el canal.

6. **El modelo asume que el chargeback rate post-servicio será bajo**. Si después de implementarnos la PyME sigue teniendo contracargos significativos (porque nosotros aprobamos casos que terminaron en fraude), el reporte va a mostrar "te ahorramos X" pero la PyME seguirá viendo el dolor en su procesador. La promesa "olvidate del problema" tiene que cumplirse en los hechos.

---

## 7. TL;DR

- **El cliente es una PyME e-commerce en Shopify/Tiendanube/WooCommerce**, no un banco.
- **El modelo es servicio gestionado**, no herramienta self-service.
- **5 momentos**: descubre el problema → integración asistida → primer reporte → operación pasiva → renueva.
- **La PyME no opera nada**. Nosotros integramos la API, el modelo decide automático, nuestro analista revisa los casos ambiguos.
- **El reporte quincenal es la experiencia del producto**: cifras agregadas + detalle por caso ambiguo con la razón del analista.
- **El analista humano es nuestro foso competitivo más fuerte** — Stripe no lo tiene, los bancos no lo escalan.
- **La PyME percibe silencio operativo**, que es exactamente lo que quería desde el momento 1.

> Para los slides 6-8 del deck final: el timeline horizontal de los 5 momentos + el mockup del reporte quincenal cubren el feedback del profesor sobre customer journey. El detalle del caso del analista en el reporte demuestra interpretabilidad del modelo (sin nombrar SHAP) y diferenciación frente a Stripe (analista humano integrado).

---

## Documentos relacionados

- [`arquitectura_realtime.md`](arquitectura_realtime.md) — la arquitectura que sostiene este servicio (Model API, Decision Engine, Case Management).
- [`kpis_financieros.md`](kpis_financieros.md) — análisis del segmento PyME y por qué tiene sentido como target.
- [`model_evaluation.md`](model_evaluation.md) — el modelo cuyas predicciones alimentan al analista.
- `mockup_reporte_quincenal_pyme` — mockup del reporte que recibe el cliente PyME (renderizado en chat).
- `mockup_panel_pyme_revision_alerta` — mockup del panel interno del analista de fraude (renderizado en chat).
- `customer_journey_pyme_servicio_gestionado` — timeline visual de los 5 momentos (renderizado en chat).

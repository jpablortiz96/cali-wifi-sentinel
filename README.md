# Cali WiFi Sentinel 360

Prototipo para la hackathon regional Cali Colombia 5.0 enfocado en el reto:

**Zonas WiFi Inteligentes: Agentes de IA para optimizar la conectividad publica**

La aplicacion se construye por pasos y evita suposiciones rigidas sobre la estructura del dataset oficial. Primero inspecciona la base real, luego permite mapear columnas y finalmente activa agentes y capas de decision alineadas con el reto oficial.

## Paso 1: Inspeccion inicial del dataset

El Paso 1 permite:

- Subir un archivo CSV, TXT o Excel.
- Guardar una copia del archivo original en `data/raw/`.
- Mostrar vista previa, columnas, tipos de datos y nulos.

## Paso 2: Configuracion de Gemini e inspeccion inteligente

El Paso 2 agrega:

- Perfil estructural avanzado del dataset.
- Deteccion heuristica de columnas candidatas.
- Guardado del perfil del dataset en `data/processed/`.
- Integracion opcional con Gemini para analizar solo el perfil estructural resumido.

Importante:

- El proyecto usa **Gemini** por medio de la libreria oficial `google-genai`.
- La variable requerida es `GEMINI_API_KEY`.
- El modelo por defecto es `gemini-2.5-flash`.
- Gemini no recibe la base completa, solo un resumen del perfil estructural.

## Paso 3: Alineacion con documento oficial Hackathon Cali

El Paso 3 alinea la app con los tres agentes del reto oficial:

- **Agente Operativo**
  - Deteccion de anomalias basicas por reglas simples.
  - Generacion preliminar de ordenes de trabajo.

- **Agente Conversacional Tecnico**
  - Preguntas en lenguaje natural sobre el rendimiento tecnico.
  - Respuestas con Gemini usando solo contexto resumido, no toda la base.

- **Agente Estrategico Geoespacial**
  - Recomendaciones de mantenimiento e inversion.
  - Uso de coordenadas reales si existen, o de territorio si no hay latitud/longitud.

## Paso 4: Inteligencia contextual y Pasaporte de Decision

El Paso 4 agrega:

- Clima contextual opcional con Open-Meteo.
- Contexto urbano opcional con OpenStreetMap Overpass.
- Variables de calendario locales.
- Indice de Impacto Ciudadano.
- Simulador de cuadrillas.
- Pasaporte de Decision.
- Cache local y bajo costo computacional.

## Paso 5: Orquestador autonomo y paquete de evidencia

El Paso 5 integra los modulos anteriores en una experiencia completa para operacion, trazabilidad y generacion de evidencia tecnica.

### Que es el ciclo autonomo

Es un flujo deterministico, sin LangChain, que ejecuta en orden:

1. Data Readiness Score
2. Variables de calendario
3. Ordenes de trabajo
4. Recomendaciones estrategicas base
5. Contexto clima opcional
6. Contexto OSM opcional
7. Indice de impacto ciudadano
8. Optimizacion de cuadrillas
9. Pasaportes de decision
10. Resumen ejecutivo base
11. Event log de agentes

Todo esto se ejecuta desde **Mission Control** al presionar el boton:

`Ejecutar ciclo autonomo`

### Que outputs genera

El ciclo autonomo produce:

- Data Readiness Score
- Contexto calendario
- Ordenes de trabajo enriquecidas
- Recomendaciones estrategicas
- Contexto clima y OSM cuando el usuario lo activa
- Impact scores por zona
- Plan de cuadrillas
- Pasaportes de decision
- Resumen ejecutivo base
- Log de eventos de agentes

### Que es el Data Readiness Score

Es un score de 0 a 100 que mide que tan util es el dataset para cumplir el reto oficial.

Evalua, entre otros:

- zona
- fecha / historico
- conexiones o uso
- trafico
- estado del punto de acceso
- latitud / longitud
- territorio
- calidad basica de datos

Ademas reporta:

- fortalezas
- brechas
- proximos pasos recomendados
- alineacion con los tres agentes oficiales

### Que es el paquete de evidencia

Es un conjunto de archivos descargables para seguimiento tecnico, operacion y toma de decisiones:

- `executive_report.md`
- `work_orders.csv`
- `impact_scores.csv`
- `crew_plan.json`
- `decision_passports.json`
- `evidence_pack.json`

Se genera desde la pestana **Paquete de Evidencia** despues de ejecutar el ciclo autonomo.

### Como usar datos sinteticos de prueba

La app incluye un modo demo con datos sinteticos opcionales para practicar antes del evento.

Importante:

- Los datos sinteticos se marcan con `tipo_dato = SINTETICO_NO_OFICIAL`.
- La interfaz muestra una alerta persistente:
  - `DATOS SINTETICOS / NO OFICIALES`
- No deben presentarse como datos reales del evento ni de la Alcaldia.

### Vista ejecutiva operativa

La aplicacion resume el estado funcional del sistema con vistas orientadas a cliente real:

- Mission Control para ejecucion integral
- Simulacion Operativa para reproducir el dataset por lotes
- Validacion Humana para revisar ordenes
- Auditoria Operativa para trazabilidad
- Paquete de Evidencia para exportar resultados legibles y descargables

### Como esto se alinea con la rubrica oficial

- **Pertinencia Territorial 25%**
  - usa territorio, geografia y contexto urbano cuando el dataset lo permite

- **Innovacion 20%**
  - integra orquestador autonomo, pasaporte de decision y paquete de evidencia

- **Viabilidad Tecnica 20%**
  - usa reglas transparentes, cache, APIs opcionales y bajo costo computacional

- **Impacto 20%**
  - traduce datos operativos en ordenes, impacto y priorizacion de cuadrillas

- **Presentacion 15%**
  - deja una experiencia guiada con Mission Control, Simulacion Operativa y Paquete de Evidencia

## Paso 6: Simulacion Operativa, Validacion Humana y Auditoria

El Paso 6 convierte el prototipo en una herramienta mas funcional para una Alcaldia, operador o cliente tecnico.

Agrega cuatro capacidades nuevas:

- **Simulacion Operativa**
  - reproduce el dataset cargado en lotes
  - si hay fecha, ordena por fecha
  - si no hay fecha, usa orden de filas y lo advierte de forma explicita
  - recalcula ordenes, impacto, cuadrillas y pasaportes en cada paso

- **Validacion Humana**
  - crea una cola de revision de ordenes de trabajo
  - permite aprobar, rechazar, marcar visita o cerrar una orden
  - deja comentario del operador y marca de tiempo

- **Blindaje Tecnico**
  - ejecuta un quality gate operativo
  - valida schema mapping, readiness operativo y outputs generados
  - explica si el sistema esta listo, limitado o bloqueado

- **Auditoria Operativa**
  - registra trazabilidad de Mission Control, simulacion y revision humana
  - permite descargar la bitacora como CSV
  - deja evidencia de que el sistema no actua como una caja negra

Texto obligatorio:

> Simulacion Operativa no significa monitoreo en tiempo real real. Es una reproduccion controlada del dataset cargado para demostrar como el sistema procesaria registros entrantes en un entorno operativo.

### Como usar Simulacion Operativa

1. Carga un CSV, TXT o Excel, o selecciona un archivo local desde `data/raw/`.
2. Ve a **Mapeo de columnas** y selecciona zona, fecha y metricas disponibles.
3. Abre **Simulacion Operativa**.
4. Define:
   - tamano de lote
   - numero de cuadrillas
5. Pulsa:
   - `Preparar simulacion`
   - `Avanzar un lote`
   - o `Ejecutar simulacion completa`

Si el dataset no tiene fecha:

- la app no se rompe
- muestra una advertencia clara
- la reproduccion usa orden de filas, no temporalidad real

### Como cargar un dataset local desde `data/raw/`

La app ahora permite dos caminos:

- subir archivo manualmente con el uploader
- o seleccionar un archivo local ya guardado en `data/raw/` o en la raiz del proyecto

Esto evita hardcodear nombres y facilita probar datasets adjuntos antes del evento.

### Archivos descargables del Paso 6

Ademas del paquete anterior, ahora pueden generarse:

- `replay_timeline.csv`
- `human_review_log.csv`
- `quality_gate_report.json`
- `operational_audit_log.csv`
- `operational_audit_summary.json`

## Limitaciones responsables del Paso 6

- La simulacion operativa no es streaming real.
- No se llaman APIs externas automaticamente.
- Si no hay fecha, no se inventa temporalidad.
- Si no hay coordenadas, el sistema sigue funcionando con limitaciones explicitas.
- Gemini no calcula scores ni prioridades; solo explica resultados.
- Los datos sinteticos siguen marcados como no oficiales.

## Paso 7: Interfaz operativa y evidencia legible

El Paso 7 limpia la interfaz para que la aplicacion se sienta mas cercana a un producto real para Alcaldia, operador o cliente tecnico.

Cambios principales:

- Se retiro la pestana **Modo Jurado**.
- El **Paquete de Evidencia** ahora muestra vistas legibles con metricas, tablas limpias, resumen ejecutivo y expanders tecnicos.
- El JSON ya no aparece como vista principal; queda como descarga o dentro de vistas avanzadas.
- Los **Pasaportes de Decision** se muestran como fichas operativas legibles.
- Las ordenes, scores, validaciones y auditoria se presentan en tablas limpias y metricas de seguimiento.

### Como generar `readable_evidence_report.md`

1. Ejecuta **Mission Control** o **Simulacion Operativa**.
2. Ve a **Paquete de Evidencia**.
3. Pulsa `Generar paquete de evidencia`.
4. La app guardara en `data/outputs/`:
   - `readable_evidence_report.md`
   - `evidence_summary.csv`
   - y el resto de archivos operativos ya existentes

### Como descargar evidencia para cliente o Alcaldia

Desde **Paquete de Evidencia** puedes descargar:

- reportes markdown legibles
- tablas CSV de ordenes, impacto, timeline y revision humana
- JSON operativos para trazabilidad tecnica
- resumen tabular `evidence_summary.csv`

Estas vistas estan pensadas para tomadores de decision, equipo tecnico y seguimiento operativo, no para mostrar estructuras crudas.

## Paso 8: Dashboard Ejecutivo 360 y diseño premium

El Paso 8 agrega una nueva vista principal llamada **Vista Ejecutiva 360** con un diseño mas cercano a un centro de comando GovTech.

Incluye:

- KPIs principales en tarjetas premium
- resumen ejecutivo automatico
- graficos interactivos tipo Power BI
- mapa geografico sin depender de token de Mapbox
- hallazgos, alertas y proximas acciones
- visualizaciones mejoradas en Impacto Ciudadano, Simulacion Operativa y Agente Estrategico

### Como usar Vista Ejecutiva 360

1. Carga datos desde uploader, archivo local o demo sintetica.
2. Ve a **Mapeo de Columnas** y completa el mapeo operativo.
3. Ejecuta **Mission Control** o **Simulacion Operativa**.
4. Abre **Vista Ejecutiva 360**.

### Que muestra la Vista Ejecutiva 360

- KPIs de registros, zonas, ordenes, readiness y confidence
- donut de clasificacion
- top zonas por impacto
- dispersion de demanda vs severidad tecnica
- radar de componentes del score
- timeline de simulacion cuando existe
- distribucion de ordenes o validacion humana
- treemap de recomendaciones
- heatmap territorial cuando existe territorio
- mapa geografico de prioridad cuando hay coordenadas

### Limitaciones responsables del dashboard

- Sin coordenadas no se activa el mapa geografico.
- Sin fecha no se activa el heatmap temporal ni el timeline real.
- Sin resultados de ciclo o simulacion no se muestra el dashboard completo.
- La vista es ejecutiva, pero no altera calculos ni reemplaza la validacion tecnica.

## Paso 8.5: Correcciones UX, mapa profesional y Vista 360 robusta

El Paso 8.5 reorganiza la navegacion para seguir el flujo real de uso del sistema:

1. Carga e Inspeccion
2. Mapeo de Columnas
3. Mission Control
4. Simulacion Operativa
5. Vista Ejecutiva 360
6. Paquete de Evidencia

### Que cambia en Vista Ejecutiva 360

- detecta resultados recientes de **Mission Control** y **Simulacion Operativa** desde `st.session_state`
- unifica ordenes, scores, pasaportes, cuadrillas, validacion humana y auditoria
- muestra un expander compacto de diagnostico de fuentes
- usa estados vacios compactos cuando aun no hay resultados
- permite generar analisis ejecutivo con Gemini solo bajo demanda

### Como usar el analisis con Gemini

1. Ejecuta **Mission Control** o **Simulacion Operativa**.
2. Abre **Vista Ejecutiva 360**.
3. Revisa el resumen deterministico.
4. Pulsa **Analizar hallazgos con Gemini** si quieres una lectura ejecutiva enriquecida.

Gemini no calcula scores ni inventa resultados. Solo analiza y redacta a partir de resultados reales ya generados por el sistema.

### Como funciona el mapa de Cali

- usa coordenadas reales del dataset si fueron mapeadas
- se centra automaticamente segun las coordenadas validas
- usa cartografia base de OpenStreetMap sin token de Mapbox
- si no hay internet, la cartografia puede no cargar; aun asi la app sigue funcionando
- si no hay latitud/longitud mapeadas, la app muestra un mensaje claro y no inventa ubicaciones

### Boton flotante y mejoras mobile

- se agrego un boton flotante **↑ Menu** para volver arriba
- las pestañas ahora permiten mejor scroll horizontal en pantallas pequenas
- se redujo padding y tamano de titulos en mobile
- las tarjetas y botones se adaptan mejor a anchos angostos

### Limitaciones responsables adicionales

- la Vista Ejecutiva 360 depende del dataset activo y del mapeo actual
- si cambias de dataset sin volver a ejecutar Mission Control o Simulacion, la vista se mantendra vacia hasta tener resultados del nuevo archivo
- el analisis con Gemini es opcional y requiere `GEMINI_API_KEY`

## APIs opcionales utilizadas

- **Open-Meteo**
  - se usa para enriquecer clima contextual por coordenada y fecha
  - aporta lluvia, temperatura, humedad y viento como contexto secundario

- **OpenStreetMap Overpass**
  - se usa para contar equipamientos cercanos como colegios, hospitales, bibliotecas, parques y transporte
  - aporta una medida de criticidad territorial aproximada

Todas las APIs externas:

- son opcionales
- usan cache local en `data/external_cache/`
- no se disparan automaticamente al cargar la app
- no rompen la app si no hay internet
- tienen limite de hasta 25 consultas por defecto

## Estructura del proyecto

```text
cali-wifi-sentinel/
|-- app.py
|-- requirements.txt
|-- README.md
|-- .gitignore
|-- .env.example
|-- data/
|   |-- raw/
|   |-- processed/
|   |-- external_cache/
|   \-- outputs/
|-- src/
|   |-- agent_orchestrator.py
|   |-- calendar_context.py
|   |-- config.py
|   |-- data_loader.py
|   |-- data_quality.py
|   |-- dashboard_insights.py
|   |-- dashboard_visuals.py
|   |-- decision_passport.py
|   |-- demo_data.py
|   |-- evidence_pack.py
|   |-- evidence_formatters.py
|   |-- external_sources.py
|   |-- gemini_client.py
|   |-- human_in_the_loop.py
|   |-- impact_scoring.py
|   |-- live_replay.py
|   |-- operational_audit.py
|   |-- osm_context.py
|   |-- profile_storage.py
|   |-- readiness_score.py
|   |-- resource_optimizer.py
|   |-- schema_mapper.py
|   |-- strategic_recommendations.py
|   |-- technical_chat.py
|   |-- ui_components.py
|   |-- utils.py
|   |-- validation_suite.py
|   |-- weather_context.py
|   \-- work_orders.py
|-- prompts/
|-- notebooks/
\-- .venv/
```

## Instalacion

### 1. Crear entorno virtual

**Windows**

```powershell
python -m venv .venv
.venv\Scripts\activate
```

**Mac/Linux**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

## Configurar Gemini con `.env`

1. Crea un archivo llamado `.env` en la raiz del proyecto.
2. Copia esta configuracion base:

```env
GEMINI_API_KEY=tu_clave_aqui
GEMINI_MODEL=gemini-2.5-flash
APP_ENV=development
```

3. Guarda el archivo.
4. Reinicia Streamlit si ya estaba corriendo.

El archivo `.env` no debe subirse a Git. Ya esta protegido por `.gitignore`.

## Como ejecutar la app

```bash
streamlit run app.py
```

## Como probar el flujo completo

1. Carga un archivo `.csv`, `.txt`, `.xlsx` o `.xls`, o usa el paquete oficial Zonas WiFi Inteligentes.
2. Revisa **Carga e inspeccion de datos**.
3. Ve a **Mapeo de columnas** y selecciona manualmente fecha, zona, conexiones, trafico, estado, latitud, longitud y territorio segun existan en la base real.
4. Revisa el **Data Readiness Score**.
5. Ve a **Mission Control** y configura:
   - numero de cuadrillas
   - usar clima si/no
   - usar OSM si/no
   - maximo de puntos externos
6. Pulsa **Ejecutar ciclo autonomo**.
7. Revisa **Vista Ejecutiva 360** para el resumen general.
8. Revisa:
   - Agente Operativo
   - Indice de Impacto Ciudadano
   - Simulador de Cuadrillas
   - Pasaporte de Decision
   - Agente Estrategico Geoespacial
   - Simulacion Operativa
   - Validacion Humana
   - Blindaje Tecnico
   - Auditoria Operativa
9. Si tienes `GEMINI_API_KEY`, usa **Agente Conversacional Tecnico**.
10. Genera el **Paquete de Evidencia**.
11. Revisa **Auditoria Operativa**, **Validacion Humana** y el **Paquete de Evidencia** como cierre del flujo.

## Limitaciones responsables

El sistema no afirma causalidad tecnica sin evidencia. El clima, los puntos de interes y el calendario se usan como variables contextuales para mejorar priorizacion, no como prueba definitiva de falla. Las recomendaciones son apoyo a la decision, no sustituyen la validacion tecnica en campo.

Los datos sinteticos quedan reservados para pruebas internas de desarrollo y no representan informacion oficial de la Alcaldia ni del evento.

## Alcance actual

La aplicacion ya esta alineada con el reto oficial, pero sigue siendo intencionalmente prudente:

- sin LangChain
- sin base de datos
- sin modelos complejos de prediccion
- sin inventar poblacion afectada
- sin inventar coordenadas
- sin afirmar fallas reales sin evidencia suficiente
- sin presentar datos sinteticos como oficiales

La meta actual es llegar al evento con una arquitectura ligera, explicable, demostrable y lista para adaptarse al dataset real en minutos.
## Paso 8.6: Mapa Cali Pro, Agente Flotante y Recomendaciones Gemini

Este ajuste corrige y amplía la experiencia ejecutiva del sistema sin cambiar los cálculos centrales. La app ahora incorpora un mapa Cali Pro más legible, un agente conversacional flotante orientado al uso de la plataforma, recomendaciones estratégicas estructuradas con Gemini y validación humana masiva.

### Qué cambia en este paso

- El mapa ejecutivo usa coordenadas reales del dataset, muestra etiquetas compactas con emoji e iniciales por zona y dibuja un límite de Cali de referencia obtenido con Overpass o, si falla, un fallback aproximado.
- Open-Meteo y OpenStreetMap/Overpass quedan activos de forma útil y controlada. Se usan desde Mission Control o desde el botón manual de enriquecimiento contextual, nunca automáticamente al cargar la app.
- Las recomendaciones estratégicas pueden generarse con Gemini a partir de resultados reales ya calculados por el sistema. Si Gemini no está configurado, la app usa un fallback determinístico.
- Se agrega un agente conversacional flotante para responder preguntas solo sobre la plataforma, la base cargada y el análisis generado.
- Validación Humana ahora permite aplicar estados y comentarios de forma masiva a todas las órdenes o a una selección.
- El botón visible de datos sintéticos se retiró de la interfaz principal. Los datos sintéticos quedan solo para pruebas internas de desarrollo.

### Uso recomendado

1. Carga un dataset real.
2. Mapea columnas.
3. Ejecuta Mission Control o Simulación Operativa.
4. Abre Vista Ejecutiva 360 para revisar KPIs, mapa y hallazgos.
5. Si hay coordenadas, activa Open-Meteo y OpenStreetMap desde Mission Control o Agente Estratégico.
6. Genera recomendaciones estratégicas para habilitar el treemap.
7. Usa Validación Humana para revisión individual o masiva.
8. Exporta evidencia operativa desde Paquete de Evidencia.

### Mapa Cali Pro

- Si hay latitud y longitud mapeadas, el mapa se centra automáticamente sobre los puntos válidos del dataset.
- Si no es posible calcular un centro con los datos, usa el centro aproximado de Cali como referencia visual.
- La cartografía base usa OpenStreetMap sin requerir token de Mapbox.
- Si no hay internet, la cartografía base puede no cargar, pero la app sigue funcionando y conserva caché cuando existe.
- Si no hay coordenadas, la plataforma muestra un mensaje claro y no inventa un mapa.

### Open-Meteo y OpenStreetMap Overpass

- Ambas fuentes usan caché en `data/external_cache/`.
- El límite por defecto de puntos externos es 20 y puede ampliarse hasta 50 con advertencia.
- No se hacen llamadas masivas automáticas.
- Si falla internet, la app intenta reutilizar caché; si no existe, muestra advertencias amables y sigue operando.

### Recomendaciones Gemini y treemap

- Las recomendaciones estratégicas con Gemini se generan solo a partir de resultados reales: readiness, quality gate, órdenes, impact scores, contexto OSM, contexto climático, pasaportes, cuadrillas, auditoría y limitaciones.
- No se envía toda la base a Gemini.
- Si Gemini devuelve salida imperfecta, la app intenta parsear JSON y, si no puede, cae a recomendaciones determinísticas.
- El treemap estratégico se alimenta con esas recomendaciones estructuradas.

### Agente conversacional flotante

El agente conversacional flotante solo responde preguntas sobre la plataforma, la base cargada y el análisis generado. No está diseñado para responder preguntas generales fuera del contexto operativo.

### Limitaciones responsables

- El sistema no inventa coordenadas ni población afectada.
- El clima y el contexto urbano se usan como variables contextuales, no como prueba definitiva de causalidad técnica.
- Sin coordenadas no hay mapa ejecutivo ni enriquecimiento geoespacial fino.
- Sin Gemini, la app sigue funcionando con fallbacks determinísticos.

## Paso 8.7: Ajustes UX finales antes del Paso 9

Este ajuste pule dos detalles clave de experiencia sin tocar los cálculos centrales:

- las etiquetas dentro de las burbujas del mapa quedaron centradas, negras y más legibles
- el agente conversacional dejó de aparecer como bloque embebido y ahora funciona como widget flotante real

### Qué cambia en los mapas

- los labels de zonas se renderizan dentro de la burbuja con mejor centrado visual
- se usa alto contraste para priorizar lectura sobre cartografía
- el estilo se unifica entre Vista Ejecutiva 360, Agente Estratégico y demás mapas geográficos de prioridad

### Qué cambia en el chat flotante

- el botón flotante de chat ahora usa un icono claro `💬`
- al hacer clic se abre un panel compacto tipo widget
- el historial queda dentro del panel, no abajo de la página
- si Gemini no está configurado, el widget sigue funcionando con ayuda contextual básica

### Convivencia entre botones flotantes

- `⬆️` sirve para volver arriba al inicio de la app
- `💬` abre el agente de plataforma
- ambos botones se mantienen separados en desktop y mobile

### Compatibilidad mobile

- el chat flotante reduce ancho y altura en pantallas angostas
- las pestañas mantienen scroll horizontal
- las tarjetas y botones conservan proporciones más compactas

El agente conversacional flotante solo responde preguntas sobre la plataforma, la base cargada y el análisis generado. No está diseñado para responder preguntas generales fuera del contexto operativo.

## Paso 8.8: Widget de chat flotante premium

En este ajuste se corrigió la experiencia visual del agente de plataforma:

- el chat ahora aparece como un botón circular flotante
- al abrirse, muestra un panel pequeño tipo widget
- ya no se incrusta como barra inferior dentro del layout
- convive con el botón `⬆️` para volver arriba
- mantiene adaptación mobile con ancho reducido y altura controlada

El botón de chat abre un panel contextual para preguntas sobre módulos, indicadores, órdenes, scores, pasaportes, validación humana, auditoría y limitaciones del análisis actual.

## Paso 9: Integraci�n con paquete oficial Zonas WiFi Inteligentes

La app ahora soporta un modo especializado para el repositorio oficial [Zonas-WiFi-Inteligentes](https://github.com/AlejandroTenorioT/Zonas-WiFi-Inteligentes). Este modo trata la fuente como un paquete multitabla Cisco Meraki, no como un CSV gen�rico.

### C�mo cargar el paquete

En **Carga e Inspecci�n** encontrar�s la secci�n **Cargar paquete oficial Zonas WiFi Inteligentes** con dos opciones:

- **Cargar desde carpeta local**: apunta a `data/raw/Zonas-WiFi-Inteligentes/` o a cualquier carpeta que contenga los CSV oficiales.
- **Cargar desde GitHub oficial**: usa la URL del repositorio para descargar los archivos esperados.

Archivos esperados:

- `network_events_curated.csv`
- `clients_curated.csv`
- `access_points_curated.csv`
- `ap_hourly_metrics_curated.csv`
- `data_dictionary.csv`

Si falta alguno, la app no se rompe: muestra advertencias y sigue operando con la mejor evidencia disponible.

### Qu� es el modo Meraki

Cuando se detecta el paquete oficial:

- se aplica un mapeo autom�tico basado en el esquema can�nico Meraki
- se habilita la construcci�n de un mart operativo por Access Point (AP)
- Mission Control y Simulaci�n Operativa cambian a un flujo horario especializado
- las �rdenes, pasaportes, scores y evidencia se interpretan por AP / zona Meraki

### Tablas que se integran

- **network_events**: eventos operativos por AP y cliente
- **clients**: estado, uso y tipo de dispositivo por cliente
- **access_points**: estado del AP, serial, IP local y `connectivity_history`
- **hourly_metrics**: conexiones, desconexiones, autenticaciones, clientes �nicos y `disconnection_rate` por hora
- **data_dictionary**: apoyo documental del paquete

### Indicadores nuevos

Al construir el mart operativo se generan, entre otros:

- `Operational Risk Score`
- `AP Health Score`
- `status_risk`
- `disconnection_risk`
- `demand_impact`
- `recurrence_risk`
- `high_disconnection_hours`
- `zero_connection_hours`
- `avg_disconnection_rate`
- `max_disconnection_rate`
- `recommended_action`

### Simulaci�n y anomal�as

- Si existe `ap_hourly_metrics_curated.csv`, la **Simulaci�n Operativa** usa `timestamp_hour` como eje principal.
- Las anomal�as Meraki comparan cada AP contra su baseline hist�rico.
- Se detectan patrones como AP offline o dormant, conexiones muy bajas frente a su propio baseline, `disconnection_rate` alta y horas sin conexiones en APs con actividad hist�rica.

### Exportables nuevos

En **Paquete de Evidencia** ahora se pueden exportar artefactos adicionales del flujo Meraki:

- `operational_mart.csv`
- `meraki_work_orders.csv`
- `meraki_anomalies.csv`
- `meraki_decision_passports.json`

### Limitaciones responsables

- El paquete no trae coordenadas exactas del AP por defecto, por lo que el mapa no debe interpretarse como ubicaci�n precisa de infraestructura.
- `connectivity_history` llega como texto exportado; sus c�digos requieren validaci�n t�cnica adicional con Cisco Meraki.
- El dataset es curado, anonimizado y no representa monitoreo en vivo.
- Si se geocodifica por nombre de zona, esa georreferenciaci�n debe tratarse como aproximada, no como ubicaci�n exacta del AP.

### C�mo interpretar los scores

- **Operational Risk Score** resume criticidad operativa por AP combinando estado, desconexiones, recurrencia y demanda.
- **AP Health Score** resume salud general; un valor bajo sugiere degradaci�n o evidencia operativa insuficiente.

## Capa Ciudadana

La plataforma ahora tiene una segunda cara orientada a usuarios finales y ciudadanía, además del backoffice operativo. Esta capa trabaja únicamente con datos agregados por AP, zona y hora; no expone `client_id`, hashes, MAC ni identificadores individuales.

Incluye:
- `Portal Ciudadano`: consulta zonas recomendadas para conectarse, alertas agregadas y mejores franjas horarias observadas.
- `Experiencia Ciudadana`: calcula `Citizen Experience Score` con base en estabilidad, disponibilidad, capacidad percibida, actividad y confianza de datos.
- `Buzón Ciudadano`: recibe reportes anónimos con calificación, problema y comentario opcional. No debe incluir datos personales.
- `Equidad Digital`: calcula un `Digital Equity Proxy` responsable para señalar señales relativas de mejora sin afirmar brechas reales ni usar población inexistente.
- `Agente Ciudadano con Gemini`: explica resultados agregados en lenguaje claro. Si Gemini no está configurado, usa fallback determinístico.

### Protección de privacidad
- No se exponen `client_id`, hashes, MAC ni identificadores de clientes.
- No se rastrean personas.
- No se infieren atributos personales.
- No se presenta geocodificación aproximada como ubicación exacta del AP.

### Limitaciones responsables
- El `Citizen Experience Score` es una aproximación agregada; no representa la experiencia exacta de cada usuario.
- El `Digital Equity Proxy` no usa población real y no debe interpretarse como brecha confirmada.
- El calendario público usa caché y solo se consulta cuando el usuario activa explícitamente el enriquecimiento.
- Sin métricas horarias suficientes, las recomendaciones de horario quedan limitadas.

## Portales por usuario final

La plataforma ahora se organiza por perfil de uso para reducir saturación visual y adaptar la experiencia al tipo de usuario:

- **Portal Técnico / Operativo**: concentra carga e inspección, Mission Control, simulación, vista ejecutiva, agente operativo, cuadrillas, validación, auditoría y evidencia.
- **Portal Ciudadano / Impacto Social**: concentra portal ciudadano, experiencia ciudadana, recomendador de zonas WiFi, buzón anónimo, equidad digital, retorno social de conectividad, agente ciudadano y vista pública de calidad.

La selección se realiza desde la barra lateral con **Selecciona el tipo de usuario**.

## Retorno Social de Conectividad

Este módulo cruza desempeño de la red WiFi con indicadores socioeconómicos agregados para priorizar mejoras donde la conectividad puede generar mayor impacto público.

### Datos socioeconómicos que acepta

- CSV / XLSX cargado manualmente
- archivo local dentro de `data/raw/`
- URL pública si el usuario la provee
- conector configurable a Socrata / [datos.gov.co](https://www.datos.gov.co)

### Fuentes agregadas soportadas

- **DANE**: IPM, NBI u otros indicadores oficiales agregados
- **SISBÉN agregado o anonimización territorial**: porcentajes por grupo o muestras anonimizadas, nunca fichas individuales
- otros datos abiertos oficiales siempre que estén agregados por zona, comuna, barrio, corregimiento, manzana o municipio

### Social ROI Connectivity Score

Se calcula con una fórmula transparente:

`social_roi_score = 0.30 * socioeconomic_vulnerability_score + 0.25 * digital_need_score + 0.20 * network_risk_score + 0.15 * citizen_potential_score + 0.10 * data_confidence_score`

Donde:

- `socioeconomic_vulnerability_score`: señal agregada construida desde IPM, NBI, desempleo, grupos SISBÉN u otros indicadores disponibles
- `digital_need_score`: combina experiencia ciudadana, proxy de equidad digital y disponibilidad
- `network_risk_score`: resume riesgo operativo observado
- `citizen_potential_score`: combina uso, actividad y criticidad territorial cuando exista
- `data_confidence_score`: mide qué tan completa y consistente es la evidencia

### Privacidad y uso responsable

- No se usan datos personales.
- No se procesan identificadores individuales de SISBÉN.
- No se infiere pobreza de personas.
- No se estigmatizan barrios o corregimientos.
- El resultado se presenta como **retorno social estimado** y no como verdad causal o diagnóstico definitivo.

### Limitaciones

- Si faltan indicadores socioeconómicos agregados, el score se degrada y la app lo declara explícitamente.
- Si no hay match territorial suficiente entre red y dataset socioeconómico, se marca como limitación.
- Sin Gemini, la explicación de retorno social usa fallback determinístico.

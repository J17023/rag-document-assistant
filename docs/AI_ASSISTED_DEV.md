# Uso de herramientas de AI-assisted development

Este documento describe cómo usé herramientas de AI durante el desarrollo, qué hice y decidí manualmente, y qué aprendí en el proceso.

## Herramientas utilizadas

| Herramienta                     | Uso                                                                                                                    |
| ------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| **Claude** (chat)               | Asistente de consulta: planeación, explicación de conceptos, propuestas de código y apoyo en el análisis de resultados |
| **GitHub Copilot** (en VS Code) | Autocompletado mientras escribía y editaba el código, y consultas puntuales dentro del editor                          |
| **VS Code**                     | Edición manual del código, revisión de cambios y ejecución                                                             |

## Cómo trabajé con la herramienta

Usé la AI como un **asistente de consulta**, no como un generador automático. El flujo de trabajo fue siempre el mismo:

1. Pedir una propuesta para **un paso a la vez** (no el proyecto completo de una vez).
2. **Aplicar los cambios manualmente** en mi editor, archivo por archivo.
3. **Ejecutar y verificar** el paso en mi equipo con un comando de prueba.
4. **Analizar la salida real** y decidir si el resultado era correcto, si había que ajustarlo o si había que descartarlo.

Cuando una parte del código no la entendía, pedía la explicación antes de usarla.

## Para qué las usé

### Claude

- **Planeación:** desglosar la prueba en pasos y definir el orden de trabajo.
- **Propuestas de código:** primera versión del script de punta a punta (_spike_) y, después, de cada módulo de la estructura final.
- **Explicación de conceptos:** qué hace cada dependencia, cómo funcionan los embeddings y las métricas de similitud, y elementos de Python (`@property`, `@dataclass`, `@lru_cache`).
- **Discusión de resultados:** contrastar mis observaciones sobre las salidas de cada ejecución para identificar la causa de los errores (recuperación o generación) y decidir qué probar a continuación. Las salidas están en [`docs/evidence/`](evidence/).
- **Estructura inicial de la documentación** (README y este documento), que completé con mis resultados, mis decisiones y los problemas que encontré.

### GitHub Copilot

- **Autocompletado al aplicar los cambios:** completar imports, firmas de funciones, type hints y estructuras repetitivas mientras escribía los módulos en el editor.
- **Docstrings y comentarios:** sugerencias iniciales que luego ajusté a mi forma de explicar cada función.
- **Comandos de prueba:** completar comandos `curl` y fragmentos de verificación en la terminal.
- **Consultas rápidas en el editor:** explicar un error o una línea concreta sin salir de VS Code.

Las sugerencias de Copilot las revisé antes de aceptarlas; varias no encajaban con la estructura del proyecto (nombres de módulos, rutas de la API) y las descarté o corregí.

## Qué hice y decidí manualmente

### Ejecución, verificación y diagnóstico

- **Ejecuté todas las versiones del sistema en mi equipo** y trabajé con las salidas reales. Ninguna decisión se tomó sin ver primero un resultado.
- **Diagnostiqué la primera versión imprimiendo los fragmentos recuperados** para cada pregunta. Así separé dos problemas distintos:
  - La pregunta del auxilio de conectividad fallaba por **recuperación**: el único fragmento con el dato mezclaba tres secciones y no quedaba entre los más similares.
  - La pregunta parcial fallaba por **generación**: el fragmento correcto llegaba, pero el modelo respondía "no encuentro".
- **Apliqué los cambios del diagnóstico a mano**, uno por uno (modelo de embeddings, tamaño de chunk, prompt), y volví a ejecutar para comprobar el efecto de cada uno.

### Experimentos y decisiones

- **Probé una regla adicional en el prompt** para combinar información de varios documentos. Ejecuté con y sin la regla: mejoraba parcialmente la pregunta multi documento, pero rompía la pregunta parcial, que ya funcionaba. **Decidí descartarla.**
- **Elegí el LLM con base en una comparación:** ejecuté la misma evaluación con qwen2.5:7b (local) y con Gemini (API): 5/6 contra 6/6. Decidí dejar el modelo configurable desde el `.env` para poder usar cualquiera de los dos.
- **Pedí hacer gratis el proyecto**, lo que llevó a usar embeddings locales, Ollama y la capa gratuita de Gemini.
- **Decidí dividir la API en routers** por funcionalidad (`health`, `documents`, `qa`), y hacerlo en un commit aparte después de verificar que la versión en un solo archivo funcionaba.
- **Pregunté si convenía cambiar la métrica de similitud** al ver que los scores no separaban bien lo relevante. Confirmé que, con embeddings normalizados, coseno, producto punto y distancia euclidiana dan el mismo orden, y que el problema es del modelo de embeddings.
- **Organicé el historial de commits**: un commit por cambio lógico, con Conventional Commits en español, y el spike conservado en el historial antes del refactor.

### Problemas que resolví durante el desarrollo

- **La pregunta del auxilio de conectividad no se respondía** aunque el dato estaba en el PDF: el fragmento correcto no se recuperaba. Lo resolví con dos cambios: el modelo de embeddings (punto siguiente) y la reducción de los chunks de 800 a 400 caracteres con 50 de solapamiento, para que cada sección quedara más aislada. La pregunta pasó a responderse correctamente.
- **Cambié el modelo de embeddings** de `paraphrase-multilingual-MiniLM-L12-v2` a `intfloat/multilingual-e5-base`. El primero está entrenado para detectar frases parecidas (paráfrasis), no para buscar el pasaje que responde una pregunta; e5 sí está entrenado para búsqueda. El cambio implicó:
  - Agregar los prefijos `query:` (preguntas) y `passage:` (documentos), que e5 necesita para funcionar bien; se aplican automáticamente solo si el modelo configurado es e5.
  - Normalizar los vectores y usar similitud coseno en Chroma.
  - Reindexar todos los documentos, porque los vectores de un modelo no son compatibles con los de otro.
  - Dejar el modelo configurable con `EMBEDDING_MODEL` en el `.env`.
- **El modelo respondía "no encuentro" ante preguntas parcialmente respondibles**, aunque tenía parte de la respuesta. Quité del prompt la instrucción de rechazo literal y agregué un ejemplo de respuesta parcial (_few-shot_). Con eso respondió la parte disponible e indicó qué faltaba.
- **El modelo local (qwen2.5:7b) fallaba en la pregunta que combina dos documentos**: omitía un auxilio y tomaba el monto equivocado, aun con los fragmentos correctos en el contexto. Probé Gemini con el mismo pipeline y el mismo prompt, y resolvió el caso (6/6 frente a 5/6). Dejé el LLM configurable desde el `.env` para elegir entre calidad (Gemini) y privacidad y costo cero (Ollama).
- `init_chat_model` fallaba porque faltaba instalar `langchain-ollama`.
- `gemini-2.5-flash` dejó de estar disponible para cuentas nuevas; actualicé el nombre del modelo según el mensaje de error.
- Gemini devolvía la respuesta como una lista de partes en vez de texto; se agregó una función para normalizarla.
- Se agotó la cuota gratuita de Gemini; cambié a Ollama para desarrollar y se agregó la opción `--delay` a la evaluación para respetar el límite por minuto.
- `langchain-community` mostraba un aviso de deprecación; se reemplazaron los loaders por una implementación propia con `pypdf`.

### Revisión del código

Revisé cada archivo al aplicarlo y pedí explicación de las partes que no entendía antes de usarlas.

**Equivalencia entre el spike y la versión modular.** Al reorganizar el código comprobé que cada módulo hiciera lo mismo que el bloque correspondiente del spike:

- El chunking modular generó los mismos 21 fragmentos que el spike.
- La recuperación devolvió los mismos fragmentos con los mismos scores.
- El prompt de la versión modular es idéntico al validado en el spike.
- Revisé el manejo de los números de página: el spike sumaba 1 al mostrar la fuente porque el loader de LangChain numeraba desde 0; en la versión modular la corrección se movió al loader propio, por lo que la etiqueta ya no debe sumar 1.

**Verificaciones de comportamiento:**

- Que reindexar dos veces no duplicara fragmentos en Chroma (21 y 21).
- Que el índice persistiera entre ejecuciones y se construyera solo cuando está vacío.
- Que `langchain-community` ya no se importara en ningún módulo antes de quitarlo de las dependencias.
- Que, tras dividir la API en routers, todos los endpoints respondieran igual y la ruta anterior (`/ingest`) devolviera 404.
- Que la validación de la API rechazara preguntas inválidas con 422.

**Preguntas de diseño que planteé y resolví:**

- Qué parámetros van en `config.py` (los valores validados en las pruebas) y cuáles en el `.env` (proveedor, modelo y claves), para no duplicar configuración.
- Si cambiar la métrica de similitud mejoraría el umbral. Con embeddings normalizados, coseno, producto punto y distancia euclidiana dan el mismo orden, así que el problema es del modelo de embeddings y no de la métrica.

### Evaluación

- Revisé cada respuesta contra el contenido de los documentos y **llené manualmente** las columnas "¿Correcta?" y "Observación" de ambos modelos.

## Qué aprendí

- **En RAG, recuperación y generación son problemas distintos.** Antes de cambiar el prompt hay que verificar si el fragmento correcto llegó al LLM. Imprimir los fragmentos recuperados fue la herramienta de diagnóstico más útil.
- **El tamaño del chunk importa.** Un dato al final de un fragmento sobre otro tema queda "enterrado": el embedding representa el tema principal y el dato no se recupera.
- **No todos los modelos de embeddings sirven para búsqueda.** Los de paráfrasis detectan frases parecidas; los modelos como e5 están entrenados para emparejar preguntas con pasajes y necesitan los prefijos `query:` y `passage:`.
- **El score de similitud sirve para ordenar, no como medida absoluta.** Con e5 los scores quedan en un rango estrecho incluso para fragmentos irrelevantes, y cambiar la métrica no lo resuelve.
- **Los modelos pequeños siguen mejor un ejemplo que una regla**, y cada ajuste del prompt puede romper otro caso. Hay que medir cada cambio con las mismas preguntas.
- **Si un fallo persiste con la recuperación correcta, el límite puede ser el modelo.** La comparación entre qwen2.5:7b y Gemini lo confirmó.
- **Validar primero y organizar después.** El script de punta a punta permitió detectar los problemas de calidad antes de invertir en la estructura modular.
- **Revisar los warnings, no solo los errores.** El aviso de que `langchain-community` estaba deprecado llevó a reemplazar los loaders.
- **La AI acelera, pero la verificación es mía.** Varias propuestas necesitaron ajustes al ejecutarlas en mi entorno (sistema operativo, versiones de librerías, nombres de modelos vigentes, límites de cuota).

## Evidencias

- [`evidence/diagnostic/`](evidence/diagnostic/): salidas de cada iteración del diagnóstico, que respaldan las decisiones técnicas.
- [`evidence/execution/`](evidence/execution/): capturas de la API, la consola y la evaluación.

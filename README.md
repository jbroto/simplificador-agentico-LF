# Simplificador de Textos con Agentes LLM
Este proyecto forma parte del Trabajo de Fin de Máster (TFM) titulado Simplificación Automática de Textos a Lectura Fácil Mediante Estrategias de Prompting. Consiste en un sistema multiagente que emplea modelos de lenguaje, a través de la API de Groq, para aproximar la simplificación de textos al estándar de Lectura Fácil.
A continuación se detallan las instrucciones necesarias para su instalación y puesta en marcha.

## 1. Obtención de la API key de Groq

El primer paso consiste en obtener acceso a la API de Groq, sobre la cual se sustenta el funcionamiento de los agentes.

1. Acceder a https://console.groq.com/
2. Registrarse o iniciar sesión con la cuenta correspondiente.
3. Acceder al panel de usuario y generar una nueva API key.
4. Guardar la clave en un lugar seguro, ya que será necesaria en los pasos posteriores.

Se recomienda no compartir esta clave ni publicarla en repositorios públicos.

## 2. Selección de un modelo compatible

No todos los modelos disponibles en Groq son adecuados para este proyecto: es necesario seleccionar uno que sea compatible con JSON Schema y Tool Use, ya que los agentes dependen de estas capacidades para estructurar sus respuestas y realizar llamadas a herramientas.

En el desarrollo de este proyecto se ha utilizado el siguiente modelo:

```
qwen/qwen3.8-27b
```

Es posible emplear un modelo distinto, siempre que cumpla con los requisitos mencionados anteriormente. Se recomienda consultar la [documentación oficial](https://console.groq.com/docs/models) de Groq para verificar la compatibilidad de cada modelo.

## 3. Configuración de las variables de entorno

Se debe crear un fichero `.env` en el directorio raíz del proyecto con el siguiente contenido:

```env
GROQ=<clave_de_api>
MODEL=<nombre_del_modelo>
```

Por ejemplo:

```env
GROQ=gsk_xxxxxxxxxxxxxxxxxxxxxxxx
MODEL=qwen/qwen3.8-27b
```

## 4. Instalación de dependencias

El proyecto emplea uv como gestor de entornos y dependencias. Para instalar los paquetes necesarios, debe ejecutarse el siguiente comando:

```bash
uv sync
```

## 5. Puesta en marcha del servidor de agentes

Una vez instaladas las dependencias, debe iniciarse el servidor encargado de gestionar los agentes:

```bash
uv run main.py
```

Este proceso debe permanecer en ejecución, ya que es el responsable de procesar las solicitudes de simplificación.

## 6. Ejecución del cliente

En una terminal distinta, sin cerrar la anterior, debe ejecutarse el cliente:

```bash
uv run cliente.py
```

El cliente permite interactuar con el sistema y solicitar la simplificación de un texto.

Los textos de entrada se encuentran en el directorio `/corpus`, donde se incluyen cinco textos de ejemplo con los que puede probarse el sistema. Para simplificar un texto propio, basta con añadirlo a dicho directorio.

## 7. Resultados

Una vez finalizado el procesamiento de un texto, los resultados se almacenan automáticamente en el directorio `/resultados`, siguiendo el siguiente formato:

- `<nombre_archivo>.txt` → texto simplificado
- `<nombre_archivo>glosario.txt` → glosario de términos correspondiente a dicho texto
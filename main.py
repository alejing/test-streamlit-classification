import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf


def main_app():
    # Título principal
    st.title("🧠 Clasificación de Tumores Cerebrales")

    # Divider
    st.markdown("---")

    st.write("Esta aplicación permite identificar el tipo de tumor cerebral mediante un modelo de clasificación "
             "entrenado con imágenes. Utilizando inteligencia artificial, el modelo clasifica las imágenes en una de "
             "cuatro categorías: **glioma**, **meningioma**, **pituitario** o **ausencia de tumor**. Esto facilita un "
             "diagnóstico preliminar rápido y preciso, asistiendo a profesionales en la identificación de distintos "
             "tipos de tumores para un mejor seguimiento y tratamiento.")

    st.warning("**Nota:** Esta predicción es solo una estimación y no representa una evaluación médica real. Maneje la "
               "información con su médico de confianza.")

    # Divider
    st.markdown("---")

    st.subheader("Por favor, carga una imagen al sistema.")

    # Load an image
    uploaded_file = st.file_uploader("Cargar una imagen ...", type="jpg")

    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    model = tf.keras.models.load_model("keras_model.h5", compile=False)

    # Load the labels
    class_names = open("labels.txt", "r").readlines()

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption=uploaded_file.name, use_column_width=True)

        # resizing the image to be at least 224x224 and then cropping from the center
        size = (224, 224)
        image = ImageOps.fit(img, size, Image.Resampling.LANCZOS)

        # turn the image into a numpy array
        image_array = np.asarray(image)

        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        # Load the image into the array
        data[0] = normalized_image_array

        # Predicts the model
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        mensaje = f"""
                #### 📊 Resultados de la Clasificación
                **📝 Clase Predicha:** {class_name[2:]}
                **📈 Puntuación de Confianza:** {(confidence_score*100):.2f}% 🔍
                """

        # Print prediction and confidence score
        st.info(mensaje)


if __name__ == '__main__':
    main_app()

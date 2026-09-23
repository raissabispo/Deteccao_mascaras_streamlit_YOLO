import os
import time
import json
import queue
from collections import deque, Counter
import av
import cv2
import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer


st.set_page_config(

    page_title="Detecção de Máscaras",
    page_icon="😷",
    layout="wide"

)


pasta_projeto = os.path.dirname(os.path.abspath(__file__))

caminho_modelo = os.path.join(

    pasta_projeto,

    "modelo",

    "best_mask_balanceado_final.pt"

)


frases_mascaras = {

    "with_mask": "Pessoa com máscara.",

    "without_mask": "Pessoa sem máscara.",

    "mask_weared_incorrect": "Pessoa usando a máscara incorretamente."

}


traducoes_mascaras = {

    "with_mask": "Pessoa com máscara",

    "without_mask": "Pessoa sem máscara",

    "mask_weared_incorrect": "Pessoa usando a máscara incorretamente"

}


CONDICAO_NAO_IDENTIFICADA = "Condição da máscara não identificada."


cores_mascaras = {

    "with_mask": (255, 120, 0),

    "mask_weared_incorrect": (0, 255, 255),

    "without_mask": (0, 0, 255)

}


if "fila_voz" not in st.session_state:

    st.session_state.fila_voz = queue.Queue()


fila_voz = st.session_state.fila_voz


@st.cache_resource

def carregar_modelo():

    return YOLO(caminho_modelo)


modelo = carregar_modelo()


class ProcessadorVideo:

    def __init__(self):

        self.historico_estados = deque(maxlen=8)

        self.estado_estavel = ()

        self.minimo_votos = 6

        self.tempo_anterior = time.time()

        self.ultimo_estado_falado = None


    def processar_frame(self, frame):

        imagem = frame.to_ndarray(

            format="bgr24"

        )


        resultados = modelo(

            imagem,

            conf=0.40,

            imgsz=320,

            verbose=False

        )


        resultado = resultados[0]


        estados_detectados = []


        if resultado.boxes is not None:

            for caixa in resultado.boxes:

                classe = int(

                    caixa.cls[0]

                )

                confianca = float(

                    caixa.conf[0]

                )

                nome_classe = modelo.names[

                    classe

                ]


                estados_detectados.append(

                    nome_classe

                )


                x1, y1, x2, y2 = map(

                    int,

                    caixa.xyxy[0]

                )


                cor = cores_mascaras.get(

                    nome_classe,

                    (255, 255, 255)

                )


                cv2.rectangle(

                    imagem,

                    (x1, y1),

                    (x2, y2),

                    cor,

                    2

                )


                nome_traduzido = (

                    traducoes_mascaras.get(

                        nome_classe,

                        nome_classe

                    )

                )


                texto = (

                    f"{nome_traduzido} "

                    f"{confianca:.0%}"

                )


                cv2.putText(

                    imagem,

                    texto,

                    (

                        x1,

                        max(

                            y1 - 10,

                            20

                        )

                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.6,

                    cor,

                    2

                )


        estado_atual = tuple(

            sorted(

                set(

                    estados_detectados

                )

            )

        )


        self.historico_estados.append(

            estado_atual

        )


        contador = Counter(

            self.historico_estados

        )


        estado_candidato, votos = (

            contador.most_common(1)[0]

            if contador

            else ((), 0)

        )


        if votos >= self.minimo_votos:

            self.estado_estavel = (

                estado_candidato

            )


        if self.estado_estavel:

            mensagens = [

                frases_mascaras.get(

                    estado,

                    CONDICAO_NAO_IDENTIFICADA

                )

                for estado in

                self.estado_estavel

            ]


            texto_estavel = " ".join(

                mensagens

            )


            if (

                self.estado_estavel

                != self.ultimo_estado_falado

            ):

                fila_voz.put(

                    texto_estavel

                )


                self.ultimo_estado_falado = (

                    self.estado_estavel

                )


            cv2.putText(

                imagem,

                texto_estavel,

                (20, 40),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.8,

                (255, 255, 255),

                2

            )


        else:

            cv2.putText(

                imagem,

                CONDICAO_NAO_IDENTIFICADA,

                (20, 40),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.8,

                (255, 255, 255),

                2

            )


        tempo_atual = time.time()


        fps = 1 / max(

            tempo_atual

            - self.tempo_anterior,

            0.001

        )


        self.tempo_anterior = (

            tempo_atual

        )


        cv2.putText(

            imagem,

            f"FPS: {fps:.1f}",

            (20, 75),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (255, 255, 255),

            2

        )


        return av.VideoFrame.from_ndarray(

            imagem,

            format="bgr24"

        )


processador = ProcessadorVideo()


def finalizar_video():

    processador.historico_estados.clear()

    processador.estado_estavel = ()

    processador.ultimo_estado_falado = None


st.title(

    "😷 Detecção de Máscaras"

)


st.write(

    "Detecção de máscaras em tempo real utilizando YOLOv8n."

)


st.markdown("---")


st.subheader(

    "Detecção em tempo real"

)


st.info(

    "Clique em START para iniciar a câmera."

)


coluna_webcam_1, coluna_webcam_2, coluna_webcam_3 = st.columns(

    [1, 2, 1]

)


with coluna_webcam_2:

    webrtc_streamer(

        key="deteccao-mascaras",

        video_frame_callback=processador.processar_frame,

        on_video_ended=finalizar_video,

        media_stream_constraints={

            "video": True,

            "audio": False

        },

        async_processing=True

    )


if not fila_voz.empty():

    try:

        mensagem = fila_voz.get_nowait()

    except queue.Empty:

        mensagem = None


    if mensagem:

        mensagem_js = json.dumps(

            mensagem,

            ensure_ascii=False

        )


        st.markdown(

            f"""

            <script>

            const texto = {mensagem_js};

            function falarTexto() {{

                if (!window.speechSynthesis) {{

                    return;

                }}

                const fala =

                    new SpeechSynthesisUtterance(

                        texto

                    );

                fala.lang = "pt-BR";

                fala.rate = 1.0;

                fala.pitch = 1.0;

                fala.volume = 1.0;


                const vozes =

                    window.speechSynthesis

                    .getVoices();


                const vozPtBr =

                    vozes.find(

                        voz =>

                            voz.lang

                            .toLowerCase()

                            === "pt-br"

                    );


                if (vozPtBr) {{

                    fala.voice = vozPtBr;

                }}


                window.speechSynthesis.cancel();


                window.speechSynthesis.speak(

                    fala

                );

            }}


            function iniciarVoz() {{

                const vozes =

                    window.speechSynthesis

                    .getVoices();


                if (vozes.length > 0) {{

                    falarTexto();

                }} else {{

                    window.speechSynthesis

                    .addEventListener(

                        "voiceschanged",

                        falarTexto,

                        {{ once: true }}

                    );

                }}

            }}


            iniciarVoz();

            </script>

            """,

            unsafe_allow_html=True

        )


st.markdown("---")


st.subheader(

    "Informações do modelo"

)


coluna1, coluna2, coluna3 = st.columns(3)


with coluna1:

    st.metric(

        "Modelo",

        "YOLOv8n"

    )


with coluna2:

    st.metric(

        "Confiança mínima",

        "0.40"

    )


with coluna3:

    st.metric(

        "Classes",

        "3"

    )


st.markdown("---")


st.subheader(

    "Classes detectadas"

)


coluna1, coluna2, coluna3 = st.columns(3)


with coluna1:

    st.markdown(

        "🟦 **with_mask**"

    )

    st.write(

        "Pessoa com máscara."

    )


with coluna2:

    st.markdown(

        "🟨 **mask_weared_incorrect**"

    )

    st.write(

        "Pessoa usando a máscara incorretamente."

    )


with coluna3:

    st.markdown(

        "🟥 **without_mask**"

    )

    st.write(

        "Pessoa sem máscara."

    )
# 📦 Importy
import os
import json
import re
import pandas as pd
import streamlit as st
from pycaret.regression import load_model, predict_model
from openai import OpenAI
from dotenv import load_dotenv
import requests
from langfuse import Langfuse
langfuse = Langfuse()

# 🌍 Załaduj zmienne środowiskowe
# load_dotenv()

# 🔍 Inicjalizacja Langfuse
# langfuse = Langfuse()

# 📥 Pobierz model z DigitalOcean Spaces
url = "https://9-mod-1-zad.fra1.digitaloceanspaces.com/model_halfmaraton.pkl"
response = requests.get(url)
if response.status_code == 200:
    with open("model_halfmaraton.pkl", "wb") as f:
        f.write(response.content)
else:
    raise Exception(f"Nie udało się pobrać modelu: {response.status_code}")

# 📦 Załaduj model
model = load_model("model_halfmaraton")

# 🖥️ Konfiguracja Streamlit
st.set_page_config(page_title="ZAPLANUJ SWÓJ MARATON", layout="centered")

# 🔐 API Key
if "api_key" not in st.session_state:
    st.session_state.api_key = ""

# 🔧 Funkcje pomocnicze
def save_api_key():
    st.session_state["api_key"] = st.session_state.input_api_key
    os.environ["OPENAI_API_KEY"] = st.session_state.api_key

def clear_input():
    st.session_state["user_input"] = ""

def extract_json(text):
    cleaned = re.sub(r"```json|```", "", text, flags=re.IGNORECASE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except:
                return None
        return None

def convert_time_to_seconds(time_str):
    try:
        h, m, s = 0, 0, 0
        parts = time_str.strip().split(':')
        if len(parts) == 2:
            m, s = map(int, parts)
        elif len(parts) == 3:
            h, m, s = map(int, parts)
        else:
            return None
        return h * 3600 + m * 60 + s
    except:
        return None

# 🧠 Logika główna z Langfuse
def calculate():
    user_input = st.session_state.user_input

    with st.spinner("⏳ Analizuję opis..."):
        if not user_input.strip():
            st.warning("Wprowadź dane w polu tekstowym")
            return

        prompt_template = (
            "Na podstawie poniższego tekstu wyodrębnij dane użytkownika: wiek (int), płeć (str: 'mężczyzna' lub 'kobieta'), "
            "czas_5km (format mm:ss lub m:ss lub hh:mm:ss). Zwróć dane w formacie JSON.\n\n"
            "Przykład: {\"wiek\": 29, \"płeć\": \"mężczyzna\", \"czas_5km\": \"25:30\"}\n\n"
            f"Tekst:\n{user_input}"
        )

        trace = langfuse.trace(name="MaratonTrace", user_id="user-001")
        span_llm = trace.span(name="OpenAI Chat", input=prompt_template)
        # ... wywołanie OpenAI ...
        span_llm.update(output=result)
        span_llm.end()

        span_pred = trace.span(name="Model Prediction", input=df.to_dict())
        # ... predykcja ...
        span_pred.update(output=prediction.to_dict())
        span_pred.end()

        trace.update(metadata={"source": "streamlit", "version": "1.0"})

        try:
            trace = langfuse.trace(name="MaratonTrace", user_id="user-001")

            # 🔍 Span: OpenAI
            span_llm = trace.span(name="OpenAI Chat", input=prompt_template)

            client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt_template}],
                model="gpt-4o"
            )

            result = response.choices[0].message.content
            span_llm.set_output(result)
            span_llm.end()

            data = extract_json(result)
            if not data:
                st.error(f"❌ {result}")
                trace.end()
                return

            brak_danych = []
            if not isinstance(data.get("wiek"), int):
                brak_danych.append("wiek")
            if data.get("płeć") not in ["mężczyzna", "kobieta"]:
                brak_danych.append("płeć")
            if not data.get("czas_5km"):
                brak_danych.append("czas_5km")

            if brak_danych:
                st.error(f"Brakuje danych: {', '.join(brak_danych)}")
                trace.end()
                return
            else:
                st.toast("✅ Wykryto dane: **wiek + płeć + czas 5 km**")

            czas_5km_total_sec = convert_time_to_seconds(data["czas_5km"])
            if czas_5km_total_sec is None:
                st.error("Nieprawidłowy format czasu.")
                trace.end()
                return

            tempo_sec = czas_5km_total_sec / 5
            df = pd.DataFrame([{
                "wiek": data["wiek"],
                "płeć_encoded": 1 if data["płeć"] == "mężczyzna" else 0,
                "tempo_sec": tempo_sec
            }])

            # 🔍 Span: PyCaret prediction
            span_pred = trace.span(name="Model Prediction", input=df.to_dict())
            prediction = predict_model(model, data=df)
            span_pred.set_output(prediction.to_dict())
            span_pred.end()

            czas = round(prediction["prediction_label"].values[0], 2)
            hours = int(czas // 3600)
            minutes = int((czas % 3600) // 60)
            seconds = int(czas % 60)
            formatted_time = f"{hours:02}:{minutes:02}:{seconds:02}"
            st.success(f"⏱️ Przewidywany czas: **{formatted_time}**")

            trace.end()

        except Exception as e:
            st.error(f"Wystąpił błąd predykcji: {e}")

# 🎛️ UI
if not st.session_state.api_key:
    st.text_input("🔑 Klucz OpenAI API", type="password", key="input_api_key")
    st.button("Zatwierdź", on_click=save_api_key)
    st.info("Aby korzystać z aplikacji wpisz swój klucz API OpenAI")
else:
    with st.container():
        st.markdown("""
            <h1 style='text-align: center; font-size: 42px; color: #f9fafb;'>ZAPLANUJ SWÓJ 🏃‍♂️ MARATON</h1>
            <hr style='border: 1px solid gray;'/>
        """, unsafe_allow_html=True)

    st.markdown("### 💬 Podaj poniżej swoje wyniki:")

    st.text_area(
        "Wpisz: wiek, płeć, swój rekord na 5 km. Zachowaj format w postaci: 28, K, 5 km w 30 minut",
        key="user_input"
    )

    st.markdown("<hr style='border: 1px solid #444;'>", unsafe_allow_html=True)

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.button("Oblicz", type="primary", on_click=calculate)

    with col6:
        st.button("Wyczyść dane", type="tertiary", on_click=clear_input)
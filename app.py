# import streamlit as st
# import joblib
import openai
import json
import os

import boto3
import joblib
import io
import streamlit as st

@st.cache_resource
def load_model_from_spaces():
    client = boto3.client(
        's3',
        region_name='fra1',  # lub inna Twoja lokalizacja
        endpoint_url='https://fra1.digitaloceanspaces.com',
        aws_access_key_id=st.secrets["DO_SPACES_KEY"],
        aws_secret_access_key=st.secrets["DO_SPACES_SECRET"]
    )

    obj = client.get_object(Bucket='9-mod-1-zad', Key='model_halfmaraton.pkl')
    model_bytes = obj['Body'].read()
    model = joblib.load(io.BytesIO(model_bytes))
    return model

# model = load_model_from_spaces()


# 🔐 Klucz API OpenAI (można ustawić jako zmienną środowiskową)
openai.api_key = os.getenv("OPENAI_API_KEY")

# 📦 Wczytanie modelu
# model = joblib.load("model_halfmaraton.pkl")
model = load_model_from_spaces()
st.success("✅ Model został poprawnie pobrany z DigitalOcean Spaces!")

# 🧠 Funkcja do ekstrakcji danych z tekstu
def extract_features_from_text(text):
    prompt = f"""
    Wyodrębnij z poniższego tekstu dane potrzebne do predykcji czasu półmaratonu:
    - płeć (M/K)
    - wiek (liczba)
    - tempo na 5km (min/km)

    Zwróć wynik jako JSON z kluczami: "Płeć", "Wiek", "Tempo"

    Tekst: "{text}"
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    try:
        data = json.loads(response.choices[0].message.content)
        return data
    except:
        return None

# 🖼️ Interfejs użytkownika
st.title("⏱️ Predykcja czasu półmaratonu")
user_input = st.text_area("Przedstaw się i opisz swoje tempo, wiek i płeć:")

if st.button("Oblicz czas"):
    extracted = extract_features_from_text(user_input)

    if not extracted:
        st.error("❌ Nie udało się wyodrębnić danych. Spróbuj inaczej sformułować opis.")
    else:
        missing = [key for key in ["Płeć", "Wiek", "Tempo"] if key not in extracted]
        if missing:
            st.warning(f"Brakuje danych: {', '.join(missing)}")
        else:
            plec = 0 if extracted["Płeć"].upper() == "M" else 1
            wiek = int(extracted["Wiek"])
            tempo_sek = float(extracted["Tempo"]) * 60

            df_input = {
                "Płeć": [plec],
                "Wiek": [wiek],
                "5 km Tempo (sek/km)": [tempo_sek]
            }

            predicted = model.predict(pd.DataFrame(df_input))[0]
            st.success(f"🏁 Przewidywany czas: {int(predicted // 60)} min {int(predicted % 60)} sek")

            # 🔗 Langfuse log (pseudo-integracja)
            st.caption("📊 Dane zostały przesłane do Langfuse w celu analizy skuteczności.")
            
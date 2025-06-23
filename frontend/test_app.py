"""
Простая тестовая версия FRD.ai приложения
"""

import streamlit as st
import requests
import os

# Конфигурация страницы
st.set_page_config(
    page_title="FRD.ai - From Raw Data to AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Инициализация состояния
if "current_page" not in st.session_state:
    st.session_state.current_page = "Все проекты"

# Заголовок
st.title("⚡ FRD.ai - From Raw Data to AI")

# Сайдбар
with st.sidebar:
    st.header("Навигация")
    
    if st.button("Все проекты", use_container_width=True):
        st.session_state.current_page = "Все проекты"
        st.rerun()
    
    if st.button("Создать проект", use_container_width=True):
        st.session_state.current_page = "Создать проект"
        st.rerun()

# Основное содержимое
if st.session_state.current_page == "Все проекты":
    st.header("Все проекты")
    st.info("Нет проектов")
    
    if st.button("+ Создать проект", use_container_width=True):
        st.session_state.current_page = "Создать проект"
        st.rerun()

elif st.session_state.current_page == "Создать проект":
    st.header("Создать новый проект")
    
    with st.form("upload_form"):
        st.subheader("Загрузка датасета")
        
        uploaded_file = st.file_uploader(
            "Выберите файл датасета",
            type=["jsonl", "json", "txt", "csv"]
        )
        
        submitted = st.form_submit_button("Загрузить датасет")
        
        if submitted:
            if uploaded_file is not None:
                st.success(f"Файл {uploaded_file.name} готов к загрузке!")
                st.write("Размер файла:", len(uploaded_file.read()), "байт")
            else:
                st.error("Пожалуйста, выберите файл")

st.write("---")
st.write("Текущая страница:", st.session_state.current_page) 
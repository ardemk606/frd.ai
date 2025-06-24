"""
FRD.ai - From Raw Data to AI
Streamlit веб-интерфейс для управления проектами генерации данных
"""

import streamlit as st
import requests
import os
import pandas as pd

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
if "selected_project_id" not in st.session_state:
    st.session_state.selected_project_id = None
if "show_generation_modal" not in st.session_state:
    st.session_state.show_generation_modal = False

# API клиент
def upload_dataset(uploaded_file, system_prompt):
    """Загрузить датасет и системный промпт в API"""
    try:
        base_url = os.getenv("API_BASE_URL", "http://localhost:7777")
        
        # Читаем содержимое файла один раз и сохраняем
        file_content = uploaded_file.read()
        
        # Проверяем что файл не пустой
        if not file_content:
            st.error("Файл пустой или не удалось прочитать содержимое")
            return None
        
        # Правильное формирование файла для requests
        files = {
            "file": (
                uploaded_file.name,
                file_content,
                uploaded_file.type or "application/octet-stream"
            )
        }
        
        # Данные формы
        data = {
            "system_prompt": system_prompt
        }
        

        
        response = requests.post(f"{base_url}/upload/dataset", files=files, data=data)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Ошибка при загрузке: {e}")
        if hasattr(e, 'response') and e.response is not None:
            st.error(f"Детали ошибки: {e.response.text}")
        return None


def get_projects():
    """Получить список всех проектов"""
    try:
        base_url = os.getenv("API_BASE_URL", "http://localhost:7777")
        response = requests.get(f"{base_url}/projects/short_info")
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Ошибка при получении проектов: {e}")
        return None


def get_project_detail(project_id):
    """Получить детальную информацию о проекте"""
    try:
        base_url = os.getenv("API_BASE_URL", "http://localhost:7777")
        response = requests.get(f"{base_url}/projects/{project_id}/detail")
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Ошибка при получении информации о проекте: {e}")
        return None


def next_step_project(project_id):
    """Перейти к следующему шагу проекта (простой переход)"""
    try:
        base_url = os.getenv("API_BASE_URL", "http://localhost:7777")
        response = requests.post(f"{base_url}/projects/{project_id}/next_step")
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Ошибка при переходе к следующему шагу: {e}")
        return None


def start_generation(project_id, examples_count, is_structured, output_format, json_schema):
    """Запустить генерацию данных"""
    try:
        base_url = os.getenv("API_BASE_URL", "http://localhost:7777")
        
        payload = {
            "project_id": project_id,
            "generation_params": {
                "examples_count": examples_count,
                "is_structured": is_structured,
                "output_format": output_format,
                "json_schema": json_schema if json_schema else None
            }
        }
        
        response = requests.post(f"{base_url}/projects/{project_id}/start_generation", json=payload)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Ошибка при запуске генерации: {e}")
        if hasattr(e, 'response') and e.response is not None:
            st.error(f"Детали ошибки: {e.response.text}")
        return None


def start_validation(project_id):
    """Запустить валидацию датасета"""
    try:
        base_url = os.getenv("API_BASE_URL", "http://localhost:7777")
        response = requests.post(f"{base_url}/dataset/{project_id}/validate")
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Ошибка при запуске валидации: {e}")
        if hasattr(e, 'response') and e.response is not None:
            st.error(f"Детали ошибки: {e.response.text}")
        return None


def start_fine_tuning(project_id):
    """Запустить LoRA fine-tuning"""
    try:
        base_url = os.getenv("API_BASE_URL", "http://localhost:7777")
        response = requests.post(f"{base_url}/projects/{project_id}/start_fine_tuning")
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Ошибка при запуске fine-tuning: {e}")
        if hasattr(e, 'response') and e.response is not None:
            st.error(f"Детали ошибки: {e.response.text}")
        return None


def show_status_pipeline(current_status):
    """Отобразить пайплайн статусов"""
    
    # Определяем пайплайн
    pipeline = [
        "NEW",
        "GENERATING_DATASET", 
        "READY_FOR_VALIDATION",
        "VALIDATION",
        "READY_FOR_FINE_TUNING",
        "FINE_TUNING",
        "READY_FOR_DEPLOY",
        "DEPLOYED"
    ]
    
    # Находим текущий индекс
    try:
        current_index = pipeline.index(current_status)
    except ValueError:
        current_index = 0
    
    # Создаём визуализацию
    st.subheader("🔄 Пайплайн проекта")
    
    cols = st.columns(len(pipeline))
    
    for i, status in enumerate(pipeline):
        with cols[i]:
            # Определяем цвет и символ
            if i < current_index:
                # Уже пройдено
                st.markdown(f"✅ **{status}**")
            elif i == current_index:
                # Текущий статус
                st.markdown(f"🔄 **{status}**")
                st.markdown("**← Сейчас здесь**")
            else:
                # Ещё не достигнуто
                st.markdown(f"⏸️ {status}")
    
    # Показываем что дальше
    if current_index < len(pipeline) - 1:
        next_status = pipeline[current_index + 1]
        st.info(f"**Следующий шаг:** {next_status}")
    else:
        st.success("**Проект завершён!** 🎉")


def show_generation_modal(project_id, project_name):
    """Отобразить модальное окно для настройки генерации"""
    
    with st.container():
        st.subheader("🚀 Настройка генерации данных")
        st.write(f"**Проект:** {project_name}")
        
        # Структурированный ответ - выносим ЗА пределы формы для динамического обновления
        is_structured = st.checkbox(
            "Структурированный ответ",
            value=False,
            help="Должен ли ответ модели быть в структурированном формате?",
            key="is_structured_checkbox"
        )
        
        # Формат вывода - также выносим для динамического обновления
        if is_structured:
            output_format = st.selectbox(
                "Формат структурированного вывода",
                options=["json"],
                index=0,
                help="Пока доступен только JSON формат",
                key="output_format_select"
            )
        else:
            st.info("Для неструктурированного ответа формат будет: text")
            output_format = "text"
        
        # JSON Schema - выносим для динамического обновления
        json_schema = None
        if is_structured:
            json_schema = st.text_area(
                "JSON Schema для валидации ответа",
                placeholder='{\n  "type": "object",\n  "properties": {\n    "answer": {"type": "string"}\n  },\n  "required": ["answer"]\n}',
                height=150,
                help="JSON Schema для валидации структурированного ответа модели",
                key="json_schema_input"
            )
        else:
            st.info("JSON Schema не требуется для неструктурированного ответа")
        
        with st.form("generation_form"):
            # Количество примеров
            examples_count = st.number_input(
                "Количество примеров для генерации",
                min_value=1,
                max_value=1000,
                value=10,
                help="От 1 до 1000 примеров"
            )
            
            # Валидация JSON Schema
            if is_structured and json_schema and json_schema.strip():
                try:
                    import json
                    json.loads(json_schema)
                    st.success("✅ JSON Schema валидна")
                except json.JSONDecodeError as e:
                    st.error(f"❌ Невалидная JSON Schema: {e}")
            
            # Кнопки действий
            col1, col2 = st.columns([1, 1])
            
            with col1:
                submitted = st.form_submit_button("🚀 Запустить генерацию", use_container_width=True, type="primary")
            
            with col2:
                cancelled = st.form_submit_button("❌ Отмена", use_container_width=True)
            
            if submitted:
                # Валидация
                if is_structured and (not json_schema or not json_schema.strip()):
                    st.error("Для структурированного ответа необходимо указать JSON Schema")
                    return None
                elif is_structured and json_schema and json_schema.strip():
                    try:
                        json.loads(json_schema)
                    except json.JSONDecodeError:
                        st.error("JSON Schema содержит ошибки")
                        return None
                
                # Запуск генерации
                with st.spinner("Запускаем генерацию..."):
                    result = start_generation(
                        project_id,
                        examples_count,
                        is_structured,
                        output_format,
                        json_schema
                    )
                    
                    if result and result.get("success"):
                        st.success(f"✅ {result.get('message')}")
                        st.info(f"🆔 ID задачи: {result.get('task_id')}")
                        st.info(f"📋 Очередь: {result.get('queue_name')}")
                        return "success"
                    else:
                        st.error("Не удалось запустить генерацию")
                        return None
            
            if cancelled:
                return "cancelled"
    
    return None


# Заголовок
st.title("⚡ FRD.ai - From Raw Data to AI")

# Сайдбар навигация
with st.sidebar:
    st.header("Навигация")
    
    # Кнопки навигации
    if st.button("Все проекты", key="nav_all", use_container_width=True, 
                 type="primary" if st.session_state.current_page == "Все проекты" else "secondary"):
        st.session_state.current_page = "Все проекты"
        st.rerun()
    
    if st.button("Создать проект", key="nav_create", use_container_width=True,
                 type="primary" if st.session_state.current_page == "Создать проект" else "secondary"):
        st.session_state.current_page = "Создать проект"
        st.rerun()
    
    # Кнопка возврата из детальной страницы
    if st.session_state.current_page == "Детали проекта":
        st.divider()
        if st.button("← Назад к проектам", key="nav_back", use_container_width=True):
            st.session_state.current_page = "Все проекты"
            st.session_state.selected_project_id = None
            st.session_state.show_generation_modal = False
            st.rerun()

# Основное содержимое страниц
if st.session_state.current_page == "Все проекты":
    st.header("Все проекты")
    
    # Кнопка обновления
    if st.button("🔄 Обновить список", key="refresh_projects"):
        st.rerun()
    
    # Загружаем проекты
    with st.spinner("Загружаем проекты..."):
        projects_data = get_projects()
    
    if projects_data and projects_data.get("success") and projects_data.get("projects"):
        projects = projects_data["projects"]
        
        st.info(f"Найдено проектов: {len(projects)}")
        
        # Отображаем список проектов
        for project in projects:
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 2, 2, 1.5])
                
                with col1:
                    st.subheader(f"📁 {project['name']}")
                    st.caption(f"ID: {project['id']}")
                
                with col2:
                    # Форматируем дату
                    created_at = project['created_at']
                    if isinstance(created_at, str):
                        # Убираем миллисекунды и Z для отображения
                        created_at = created_at.replace('T', ' ').split('.')[0]
                    st.write(f"**Создан:**")
                    st.write(f"{created_at}")
                
                with col3:
                    # Цвет статуса
                    status = project['status']
                    if status == 'NEW':
                        st.write(f"🟢 {status}")
                    elif status == 'PROCESSING':
                        st.write(f"🟡 {status}")
                    elif status == 'COMPLETED':
                        st.write(f"🔵 {status}")
                    else:
                        st.write(f"🔴 {status}")
                
                with col4:
                    if st.button(
                        "📂 Открыть", 
                        key=f"open_{project['id']}", 
                        help="Открыть проект",
                        use_container_width=True,
                        type="primary"
                    ):
                        st.session_state.selected_project_id = project['id']
                        st.session_state.current_page = "Детали проекта"
                        st.rerun()
                
                st.divider()
    else:
        st.info("Нет проектов")
        
        # Кнопка создания проекта
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("+ Создать проект", key="create_from_main", use_container_width=True):
                st.session_state.current_page = "Создать проект"
                st.rerun()

elif st.session_state.current_page == "Создать проект":
    st.header("Создать новый проект")
    
    # Инициализация состояния успешной загрузки
    if "upload_success" not in st.session_state:
        st.session_state.upload_success = False
        st.session_state.upload_result = None
    
    # Форма загрузки файла
    with st.form("upload_form", clear_on_submit=True):
        st.subheader("Загрузка датасета")
        
        uploaded_file = st.file_uploader(
            "Выберите файл датасета",
            type=["jsonl", "json", "txt", "csv"],
            help="Поддерживаемые форматы: JSONL, JSON, TXT, CSV"
        )
        
        st.subheader("Системный промпт")
        
        system_prompt = st.text_area(
            "Введите системный промпт для обучения модели",
            placeholder="Например: Ты дружелюбный ассистент, который помогает пользователям с их вопросами...",
            height=150,
            help="Системный промпт определяет поведение и стиль ответов модели"
        )
        
        submitted = st.form_submit_button("Создать проект", use_container_width=True)
        
        if submitted:
            if uploaded_file is None:
                st.error("Пожалуйста, выберите файл датасета")
            elif not system_prompt.strip():
                st.error("Пожалуйста, введите системный промпт")
            else:
                with st.spinner("Создаём проект..."):
                    result = upload_dataset(uploaded_file, system_prompt.strip())
                    
                    if result and result.get("success"):
                        st.session_state.upload_success = True
                        st.session_state.upload_result = result
                        st.rerun()
    
    # Отображение результата загрузки вне формы
    if st.session_state.upload_success and st.session_state.upload_result:
        st.success(f"Успешно: {st.session_state.upload_result.get('message')}")
        
        result_data = {
            "ID проекта": st.session_state.upload_result.get('dataset_id'),
            "Датасет в хранилище": st.session_state.upload_result.get('object_name')
        }
        
        # Добавляем информацию о системном промпте если есть
        if st.session_state.upload_result.get('system_prompt_object_name'):
            result_data["Системный промпт в хранилище"] = st.session_state.upload_result.get('system_prompt_object_name')
        
        st.json(result_data)
        
        # Кнопка перехода к проектам (вне формы)
        if st.button("Перейти к проектам", key="goto_projects"):
            st.session_state.current_page = "Все проекты"
            st.session_state.upload_success = False
            st.session_state.upload_result = None
            st.rerun()

elif st.session_state.current_page == "Детали проекта":
    if st.session_state.selected_project_id is None:
        st.error("Проект не выбран")
        st.session_state.current_page = "Все проекты"
        st.rerun()
    
    # Получаем детальную информацию
    with st.spinner("Загружаем информацию о проекте..."):
        project_detail = get_project_detail(st.session_state.selected_project_id)
    
    if project_detail and project_detail.get("success"):
        project = project_detail["project"]
        
        # Заголовок страницы
        st.header(f"📁 {project['name']}")
        st.caption(f"ID: {project['id']} | Создан: {project['created_at']}")
        
        # Отображаем пайплайн
        show_status_pipeline(project['status'])
        
        # Системный промпт
        st.divider()
        st.subheader("📜 Системный промпт")
        if project['system_prompt']:
            with st.expander("Показать системный промпт", expanded=False):
                st.text_area(
                    "Содержимое:", 
                    value=project['system_prompt'], 
                    height=200,
                    disabled=True,
                    label_visibility="collapsed"
                )
        else:
            st.info("Системный промпт не загружен")
        
        # Превью датасета
        st.divider()
        st.subheader("📊 Превью датасета (первые 5 строк)")
        
        if project['dataset_preview']:
            # Проверяем, есть ли ошибка
            if len(project['dataset_preview']) == 1 and 'error' in project['dataset_preview'][0]:
                st.error(project['dataset_preview'][0]['error'])
            else:
                # Создаём DataFrame для отображения
                try:
                    df = pd.DataFrame(project['dataset_preview'])
                    st.dataframe(df, use_container_width=True)
                except Exception as e:
                    st.error(f"Ошибка при отображении таблицы: {e}")
                    
                    # Показываем как JSON если таблица не получается
                    with st.expander("Показать как JSON"):
                        for i, item in enumerate(project['dataset_preview'], 1):
                            st.json(item)
        else:
            st.info("Нет данных для превью")
        
        # Действия с проектом
        st.divider()
        
        # Проверяем, не в финальном ли статусе
        if project['status'] != 'DEPLOYED':
            # Если статус NEW - показываем специальную кнопку для генерации
            if project['status'] == 'NEW':
                if st.button("🚀 Настроить генерацию", key="setup_generation", use_container_width=True, type="primary"):
                    st.session_state.show_generation_modal = True
                    st.rerun()
            
            # Если статус READY_FOR_VALIDATION - показываем кнопку валидации
            elif project['status'] == 'READY_FOR_VALIDATION':
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("🔍 Запустить валидацию", key="start_validation", use_container_width=True, type="primary"):
                        with st.spinner("Запускаем валидацию датасета..."):
                            result = start_validation(st.session_state.selected_project_id)
                            
                            if result and result.get("success"):
                                st.success(f"✅ {result.get('message')}")
                                st.info(f"🆔 ID задачи: {result.get('task_id')}")
                                st.rerun()  # Обновляем страницу чтобы показать новый статус
                            else:
                                st.error("Не удалось запустить валидацию")
                
                with col2:
                    if st.button("⏭️ Пропустить валидацию", key="skip_validation", use_container_width=True):
                        with st.spinner("Переводим проект к следующему шагу..."):
                            result = next_step_project(st.session_state.selected_project_id)
                            
                            if result and result.get("success"):
                                st.success(f"✅ {result.get('message')}")
                                st.rerun()  # Обновляем страницу чтобы показать новый статус
                            else:
                                st.error("Не удалось перейти к следующему шагу")
            
            # Если статус READY_FOR_FINE_TUNING - показываем кнопку fine-tuning
            elif project['status'] == 'READY_FOR_FINE_TUNING':
                if st.button("🔥 Запустить LoRA Fine-tuning", key="start_fine_tuning", use_container_width=True, type="primary"):
                    with st.spinner("Запускаем LoRA дообучение..."):
                        result = start_fine_tuning(st.session_state.selected_project_id)
                        
                        if result and result.get("success"):
                            st.success(f"✅ {result.get('message')}")
                            st.info(f"🆔 ID задачи: {result.get('task_id')}")
                            st.info(f"📋 Очередь: {result.get('queue_name')}")
                            st.rerun()  # Обновляем страницу чтобы показать новый статус
                        else:
                            st.error("Не удалось запустить fine-tuning")
            
            else:
                # Для остальных статусов - простой переход
                if st.button("▶️ Следующий шаг", key="next_step", use_container_width=True, type="primary"):
                    with st.spinner("Переводим проект к следующему шагу..."):
                        result = next_step_project(st.session_state.selected_project_id)
                        
                        if result and result.get("success"):
                            st.success(f"✅ {result.get('message')}")
                            st.rerun()  # Обновляем страницу чтобы показать новый статус
                        else:
                            st.error("Не удалось перейти к следующему шагу")
        else:
            st.success("🎉 Проект завершён!")
        
        # Модальное окно для генерации
        if st.session_state.show_generation_modal:
            st.divider()
            modal_result = show_generation_modal(project['id'], project['name'])
            
            if modal_result == "success":
                st.session_state.show_generation_modal = False
                st.rerun()  # Обновляем страницу чтобы показать новый статус
            elif modal_result == "cancelled":
                st.session_state.show_generation_modal = False
                st.rerun()
    
    else:
        st.error("Не удалось загрузить информацию о проекте")

# Отладочная информация (можно убрать позже)
st.sidebar.write("---")
st.sidebar.write("Отладка:")
st.sidebar.write(f"Текущая страница: {st.session_state.current_page}")
if st.session_state.selected_project_id:
    st.sidebar.write(f"Выбранный проект: {st.session_state.selected_project_id}") 
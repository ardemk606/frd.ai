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
    return _make_api_request(
        "/projects/short_info",
        "Ошибка при получении проектов",
        method="GET"
    )


def get_project_detail(project_id):
    """Получить детальную информацию о проекте"""
    return _make_api_request(
        f"/projects/{project_id}/detail",
        "Ошибка при получении информации о проекте",
        method="GET"
    )


def next_step_project(project_id):
    """Перейти к следующему шагу проекта (простой переход)"""
    return _make_api_request(
        f"/projects/{project_id}/next_step",
        "Ошибка при переходе к следующему шагу"
    )


def start_generation(project_id, examples_count, is_structured, output_format, json_schema, model_id=None):
    """Запустить генерацию данных"""
    try:
        base_url = os.getenv("API_BASE_URL", "http://localhost:7777")
        
        payload = {
            "project_id": project_id,
            "generation_params": {
                "examples_count": examples_count,
                "is_structured": is_structured,
                "output_format": output_format,
                "json_schema": json_schema if json_schema else None,
                "model_id": model_id
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
    return _make_api_request(
        f"/dataset/{project_id}/validate",
        "Ошибка при запуске валидации"
    )


def _make_api_request(endpoint_path, error_message_prefix, method="POST", **kwargs):
    """Приватный метод для выполнения запросов к API"""
    try:
        base_url = os.getenv("API_BASE_URL", "http://localhost:7777")
        
        if method.upper() == "GET":
            response = requests.get(f"{base_url}{endpoint_path}", **kwargs)
        elif method.upper() == "POST":
            response = requests.post(f"{base_url}{endpoint_path}", **kwargs)
        else:
            raise ValueError(f"Неподдерживаемый HTTP метод: {method}")
            
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"{error_message_prefix}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            st.error(f"Детали ошибки: {e.response.text}")
        return None


def skip_generation(project_id):
    """Пропустить генерацию и перейти сразу к валидации"""
    return _make_api_request(
        f"/projects/{project_id}/skip_generation",
        "Ошибка при пропуске генерации"
    )


def skip_validation(project_id):
    """Пропустить валидацию и перейти сразу к fine-tuning"""
    return _make_api_request(
        f"/projects/{project_id}/skip_validation", 
        "Ошибка при пропуске валидации"
    )


def start_fine_tuning(project_id):
    """Запустить LoRA fine-tuning"""
    return _make_api_request(
        f"/projects/{project_id}/start_fine_tuning",
        "Ошибка при запуске fine-tuning"
    )


def get_available_models():
    """Получить список доступных моделей LLM"""
    return _make_api_request(
        "/models/available",
        "Ошибка при получении списка моделей",
        method="GET"
    )


def get_default_model():
    """Получить модель по умолчанию"""
    return _make_api_request(
        "/models/default",
        "Ошибка при получении модели по умолчанию",
        method="GET"
    )


def get_available_lora_adapters():
    """Получить список доступных LoRA-адаптеров"""
    return _make_api_request(
        "/playground/adapters",
        "Ошибка при получении списка LoRA-адаптеров",
        method="GET"
    )


def get_playground_models():
    """Получить список доступных моделей для плейграунда"""
    return _make_api_request(
        "/playground/models",
        "Ошибка при получении списка моделей",
        method="GET"
    )


def stream_inference(messages, adapter_id=None, system_prompt=None, max_tokens=512, temperature=0.7, top_k=50, top_p=0.9):
    """Запустить стриминговый inference"""
    try:
        import requests
        import json
        
        base_url = os.getenv("API_BASE_URL", "http://localhost:7777")
        
        payload = {
            "messages": [{"role": msg["role"], "content": msg["content"]} for msg in messages],
            "adapter_id": adapter_id,
            "system_prompt": system_prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p
        }
        
        response = requests.post(
            f"{base_url}/playground/inference/stream", 
            json=payload,
            stream=True,
            headers={'Accept': 'text/plain'},
            timeout=60
        )
        response.raise_for_status()
        
        # Генератор для стриминга
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    try:
                        data = json.loads(line_str[6:])  # Убираем "data: "
                        yield data
                    except json.JSONDecodeError:
                        continue
                        
    except Exception as e:
        yield {"type": "error", "content": f"Ошибка при стриминге: {str(e)}"}


def show_status_pipeline(current_status):
    """Отобразить пайплайн статусов"""
    
    # Определяем пайплайн
    pipeline = [
        "NEW",
        "GENERATING_DATASET", 
        "READY_FOR_VALIDATION",
        "VALIDATING",
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
        
        # Выбор модели - выносим ЗА пределы формы для динамического обновления
        st.subheader("🤖 Выбор модели")
        
        # Загружаем доступные модели
        models_data = get_available_models()
        selected_model_id = None
        
        if models_data and models_data.get("success") and models_data.get("models"):
            models = models_data["models"]
            
            # Находим модель по умолчанию
            default_model_index = 0
            for i, model in enumerate(models):
                if model.get("is_default", False):
                    default_model_index = i
                    break
            
            # Создаем список опций для selectbox
            model_options = []
            for model in models:
                display_text = f"{model['display_name']} ({model['model_id']})"
                if model.get("is_default", False):
                    display_text += " [По умолчанию]"
                model_options.append(display_text)
            
            selected_option = st.selectbox(
                "Модель для генерации",
                options=model_options,
                index=default_model_index,
                help="Выберите модель LLM для генерации данных",
                key="model_selection"
            )
            
            # Извлекаем model_id из выбранной опции
            selected_index = model_options.index(selected_option)
            selected_model_id = models[selected_index]["model_id"]
            
            # Показываем описание выбранной модели
            selected_model = models[selected_index]
            if selected_model.get("description"):
                st.info(f"📝 {selected_model['description']}")
        else:
            st.warning("⚠️ Не удалось загрузить список моделей. Будет использована модель по умолчанию.")
        
        st.divider()
        
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
                        json_schema,
                        model_id=selected_model_id
                    )
                    
                    if result and result.get("success"):
                        st.success(f"✅ {result.get('message')}")
                        st.info(f"🆔 ID задачи: {result.get('task_id')}")
                        st.info(f"📋 Очередь: {result.get('queue_name')}")
                        
                        # Показываем информацию о выбранной модели
                        if result.get('model_id'):
                            st.info(f"🤖 Модель: {result.get('model_id')}")
                        elif selected_model_id:
                            st.info(f"🤖 Модель: {selected_model_id}")
                        else:
                            st.info("🤖 Модель: по умолчанию")
                        
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
    
    if st.button("🎮 Плейграунд", key="nav_playground", use_container_width=True,
                 type="primary" if st.session_state.current_page == "Плейграунд" else "secondary"):
        st.session_state.current_page = "Плейграунд"
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
            # Если статус NEW - показываем кнопки для генерации или пропуска
            if project['status'] == 'NEW':
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("🚀 Настроить генерацию", key="setup_generation", use_container_width=True, type="primary"):
                        st.session_state.show_generation_modal = True
                        st.rerun()
                
                with col2:
                    if st.button("⏩ Пропустить генерацию", key="skip_generation", use_container_width=True):
                        with st.spinner("Пропускаем генерацию и переходим к валидации..."):
                            result = skip_generation(st.session_state.selected_project_id)
                            
                            if result and result.get("success"):
                                st.success(f"✅ {result.get('message')}")
                                if result.get('seed_records_count'):
                                    st.info(f"📊 Использовано {result.get('seed_records_count')} записей из исходного датасета")
                                st.rerun()  # Обновляем страницу чтобы показать новый статус
                            else:
                                st.error("Не удалось пропустить генерацию")
            
            # Если статус GENERATING_DATASET - показываем информацию о процессе
            elif project['status'] == 'GENERATING_DATASET':
                st.info("🔄 Генерация данных выполняется...")
                st.caption("Генерация продолжится в фоне. Страница обновится автоматически при завершении.")
                
                # Кнопка для ручного обновления статуса
                if st.button("🔄 Обновить статус", key="refresh_status", use_container_width=True):
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
                    if st.button("⏭️ Пропустить валидацию", key="skip_validation_btn", use_container_width=True):
                        with st.spinner("Пропускаем валидацию и переходим к fine-tuning..."):
                            result = skip_validation(st.session_state.selected_project_id)
                            
                            if result and result.get("success"):
                                st.success(f"✅ {result.get('message')}")
                                st.rerun()  # Обновляем страницу чтобы показать новый статус
                            else:
                                st.error("Не удалось пропустить валидацию")
            
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

elif st.session_state.current_page == "Плейграунд":
    st.header("🎮 Плейграунд LoRA-адаптеров")
    st.caption("Тестируйте свои обученные LoRA-адаптеры в режиме реального времени")
    
    # Инициализация состояния плейграунда
    if "playground_messages" not in st.session_state:
        st.session_state.playground_messages = []
    if "playground_current_adapter" not in st.session_state:
        st.session_state.playground_current_adapter = None
    
    # Левая колонка - настройки
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("⚙️ Настройки")
        
        # Выбор LoRA-адаптера
        st.write("**LoRA-адаптер:**")
        adapters_data = get_available_lora_adapters()
        
        if adapters_data and adapters_data.get("success") and adapters_data.get("adapters"):
            adapters = adapters_data["adapters"]
            
            # Добавляем опцию "Без адаптера" (базовая модель)
            adapter_options = ["Без адаптера (базовая модель)"]
            adapter_options.extend([f"{adapter['name']} (ID: {adapter['id']})" for adapter in adapters])
            
            selected_adapter_option = st.selectbox(
                "Выберите адаптер:",
                options=adapter_options,
                key="adapter_select"
            )
            
            # Определяем выбранный адаптер
            if selected_adapter_option == "Без адаптера (базовая модель)":
                selected_adapter = None
            else:
                # Извлекаем ID из строки
                adapter_id = int(selected_adapter_option.split("ID: ")[1].split(")")[0])
                selected_adapter = next((a for a in adapters if a["id"] == adapter_id), None)
        else:
            st.warning("Нет доступных LoRA-адаптеров")
            selected_adapter = None
        
        # Системный промпт
        st.write("**Системный промпт:**")
        system_prompt = st.text_area(
            "Введите системный промпт:",
            value="Ты полезный ассистент, который отвечает на вопросы пользователей.",
            height=100,
            key="playground_system_prompt"
        )
        
        # Параметры генерации
        st.write("**Параметры генерации:**")
        
        max_tokens = st.slider(
            "Max tokens:",
            min_value=50,
            max_value=2048,
            value=512,
            step=50,
            key="playground_max_tokens"
        )
        
        temperature = st.slider(
            "Temperature:",
            min_value=0.1,
            max_value=2.0,
            value=0.7,
            step=0.1,
            key="playground_temperature"
        )
        
        top_k = st.slider(
            "Top K:",
            min_value=1,
            max_value=100,
            value=50,
            step=1,
            key="playground_top_k"
        )
        
        top_p = st.slider(
            "Top P:",
            min_value=0.1,
            max_value=1.0,
            value=0.9,
            step=0.05,
            key="playground_top_p"
        )
        
        # Кнопка очистки истории
        if st.button("🗑️ Очистить историю", use_container_width=True):
            st.session_state.playground_messages = []
            st.rerun()
    
    with col2:
        st.subheader("💬 Чат")
        
        # Показываем информацию о текущем адаптере
        if selected_adapter:
            st.info(f"🤖 Активный адаптер: {selected_adapter['name']} (Датасет: {selected_adapter['dataset_name']})")
        else:
            st.info("🤖 Активна базовая модель (без LoRA-адаптера)")
        
        # Отображение истории сообщений
        message_container = st.container()
        
        with message_container:
            for i, message in enumerate(st.session_state.playground_messages):
                if message["role"] == "user":
                    with st.chat_message("user"):
                        st.write(message["content"])
                else:
                    with st.chat_message("assistant"):
                        st.write(message["content"])
        
        # Поле ввода сообщения
        with st.container():
            user_input = st.chat_input("Введите ваш вопрос...")
            
            if user_input:
                # Добавляем сообщение пользователя
                st.session_state.playground_messages.append({
                    "role": "user",
                    "content": user_input
                })
                
                # Показываем сообщение пользователя сразу
                with st.chat_message("user"):
                    st.write(user_input)
                
                # Начинаем генерацию ответа
                with st.chat_message("assistant"):
                    response_placeholder = st.empty()
                    
                    # Реальный стриминговый inference
                    try:
                        # Определяем ID адаптера
                        adapter_id = selected_adapter["id"] if selected_adapter else None
                        
                        # Запускаем стриминг
                        streamed_text = ""
                        for data in stream_inference(
                            messages=st.session_state.playground_messages + [{"role": "user", "content": user_input}],
                            adapter_id=adapter_id,
                            system_prompt=system_prompt,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            top_k=top_k,
                            top_p=top_p
                        ):
                            if data.get("type") == "token":
                                streamed_text += data.get("content", "")
                                response_placeholder.write(streamed_text)
                            elif data.get("type") == "error":
                                st.error(f"Ошибка генерации: {data.get('content')}")
                                streamed_text = "Произошла ошибка при генерации ответа."
                                break
                            elif data.get("type") == "done":
                                # Используем полный текст из done сообщения если он есть
                                final_text = data.get("content", streamed_text)
                                if final_text and final_text != streamed_text:
                                    streamed_text = final_text
                                    response_placeholder.write(streamed_text)
                                break
                        
                        # Добавляем ответ в историю
                        st.session_state.playground_messages.append({
                            "role": "assistant", 
                            "content": streamed_text
                        })
                        
                    except Exception as e:
                        error_message = f"Ошибка подключения к inference worker: {str(e)}"
                        st.error(error_message)
                        response_placeholder.write(error_message)
                        
                        # Добавляем ошибку в историю
                        st.session_state.playground_messages.append({
                            "role": "assistant", 
                            "content": error_message
                        })
        
        # Информация о статусе
        if selected_adapter:
            st.success(f"✅ Готов к работе с LoRA-адаптером: {selected_adapter['name']}")
        else:
            st.success("✅ Готов к работе с базовой моделью")

# Отладочная информация (можно убрать позже)
st.sidebar.write("---")
st.sidebar.write("Отладка:")
st.sidebar.write(f"Текущая страница: {st.session_state.current_page}")
if st.session_state.selected_project_id:
    st.sidebar.write(f"Выбранный проект: {st.session_state.selected_project_id}") 
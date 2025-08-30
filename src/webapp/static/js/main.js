// ============================================================================
// КЛАСС ДЛЯ УПРАВЛЕНИЯ СОСТОЯНИЕМ ПРИЛОЖЕНИЯ
// ============================================================================

class AppState {
    constructor() {
        this.file = null;
        this.isUploading = false;
        this.isProcessing = false;
        this.taskId = null;
        this.socket = null;
    }

    setFile(file) {
        this.file = file;
        this.updateUI();
    }

    clearFile() {
        this.file = null;
        this.updateUI();
    }

    setUploading(uploading) {
        this.isUploading = uploading;
        this.updateUI();
    }

    setProcessing(processing) {
        this.isProcessing = processing;
        this.updateUI();
    }

    setTaskId(taskId) {
        this.taskId = taskId;
        if (taskId) {
            this.initializeSocket();
        }
    }

    updateUI() {
        // Обновляем состояние кнопки
        const submitBtn = document.getElementById('submit-btn');
        if (submitBtn) {
            const canSubmit = this.file && !this.isUploading && !this.isProcessing;
            submitBtn.disabled = !canSubmit;
            submitBtn.textContent = this.isUploading ? 'Загрузка...' : 'Загрузить и обработать видео';
        }

        // Обновляем видимость элементов
        this.updateElementVisibility();
    }

    updateElementVisibility() {
        const fileDropZone = document.getElementById('file-drop-zone');
        const fileInfo = document.getElementById('file-info');
        const notification = document.getElementById('upload-notification');

        // Дроп зона скрыта по умолчанию, показывается только при явном запросе
        if (fileDropZone) {
            fileDropZone.style.display = 'none';
        }

        // Показываем информацию о файле только если есть файл И не идет загрузка
        if (fileInfo) {
            const shouldShowFileInfo = this.file && !this.isUploading;
            fileInfo.style.display = shouldShowFileInfo ? 'block' : 'none';
        }

        if (notification) {
            notification.style.display = this.isUploading ? 'block' : 'none';
        }
    }

    showDropZone() {
        const fileDropZone = document.getElementById('file-drop-zone');
        if (fileDropZone) {
            fileDropZone.style.display = 'block';
        }
    }

    initializeSocket() {
        if (!this.taskId) return;

        this.socket = io();
        this.socket.emit("join", { task_id: this.taskId });

        this.socket.on("progress_update", (data) => {
            if (data.task_id === this.taskId) {
                UI.updateProgress(data.stage, data.progress, data.message);
            }
        });

        this.socket.on("processing_complete", (data) => {
            if (data.task_id === this.taskId) {
                this.setProcessing(false);
                UI.showResults(data.results);
            }
        });

        this.socket.on("processing_error", (data) => {
            if (data.task_id === this.taskId) {
                this.setProcessing(false);
                UI.showError(data.error);
            }
        });

        this.socket.on("joined", (data) => {
            console.log("Присоединился к комнате:", data.task_id);
        });
    }
}

// ============================================================================
// КЛАСС ДЛЯ УПРАВЛЕНИЯ ПОЛЬЗОВАТЕЛЬСКИМ ИНТЕРФЕЙСОМ
// ============================================================================

class UI {
    static elements = {};

    static initialize() {
        // Инициализируем все элементы UI
        this.elements = {
            fileInput: document.getElementById('file-input'),
            fileDropZone: document.getElementById('file-drop-zone'),
            fileInfo: document.getElementById('file-info'),
            fileName: document.getElementById('file-name'),
            fileSize: document.getElementById('file-size'),
            removeFileBtn: document.getElementById('remove-file'),
            submitBtn: document.getElementById('submit-btn'),
            uploadForm: document.getElementById('upload-form'),
            notification: document.getElementById('upload-notification'),
            progressContainer: document.getElementById('progress-container'),
            progressStage: document.getElementById('progress-stage'),
            progressPercentage: document.getElementById('progress-percentage'),
            progressBar: document.getElementById('progress-bar'),
            progressMessage: document.getElementById('progress-message'),
            resultsDiv: document.getElementById('results'),
            containerDiv: document.getElementById('container')
        };

        this.setupEventListeners();
        this.initializeFromExistingTask();
    }

    static initializeFromExistingTask() {
        const taskIdElement = document.querySelector('[data-task-id]');
        if (taskIdElement) {
            const taskId = taskIdElement.getAttribute('data-task-id');
            appState.setTaskId(taskId);
            appState.setProcessing(true);
        }
    }

    static setupEventListeners() {
        // Обработчики файлов
        if (this.elements.fileInput) {
            this.elements.fileInput.addEventListener('change', this.handleFileSelect.bind(this));
        }

        if (this.elements.removeFileBtn) {
            this.elements.removeFileBtn.addEventListener('click', this.handleRemoveFile.bind(this));
        }

        // Drag & Drop
        if (this.elements.fileDropZone) {
            this.elements.fileDropZone.addEventListener('dragover', this.handleDragOver.bind(this));
            this.elements.fileDropZone.addEventListener('dragenter', this.handleDragEnter.bind(this));
            this.elements.fileDropZone.addEventListener('dragleave', this.handleDragLeave.bind(this));
            this.elements.fileDropZone.addEventListener('drop', this.handleDrop.bind(this));
        }

        // Форма
        if (this.elements.uploadForm) {
            this.elements.uploadForm.addEventListener('submit', this.handleFormSubmit.bind(this));
        }
    }

    // ============================================================================
    // ОБРАБОТЧИКИ СОБЫТИЙ
    // ============================================================================

    static handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            if (this.isValidVideoFile(file)) {
                this.showFileInfo(file);
                appState.setFile(file);
            } else {
                alert("Пожалуйста, выберите видеофайл (MP4, MOV, AVI)");
                this.resetFileSelection();
            }
        }
    }

    static handleRemoveFile(e) {
        e.preventDefault();
        this.resetFileSelection();
        appState.clearFile();
    }

    static handleDragOver(e) {
        e.preventDefault();
        this.addDragEffects();
    }

    static handleDragEnter(e) {
        e.preventDefault();
        this.addDragEffects();
    }

    static handleDragLeave(e) {
        e.preventDefault();
        if (!this.elements.fileDropZone.contains(e.relatedTarget)) {
            this.removeDragEffects();
        }
    }

    static handleDrop(e) {
        e.preventDefault();
        this.removeDragEffects();

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            const file = files[0];
            if (this.isValidVideoFile(file)) {
                this.elements.fileInput.files = files;
                this.showFileInfo(file);
                appState.setFile(file);
            } else {
                alert("Пожалуйста, выберите видеофайл (MP4, MOV, AVI)");
            }
        }
    }

    static handleFormSubmit(e) {
        if (!appState.file) {
            e.preventDefault();
            alert("Пожалуйста, выберите видеофайл");
            return;
        }

        appState.setUploading(true);
        // Форма отправится автоматически
    }

    // ============================================================================
    // МЕТОДЫ ДЛЯ РАБОТЫ С ФАЙЛАМИ
    // ============================================================================

    static showFileInfo(file) {
        if (!this.elements.fileName || !this.elements.fileSize) return;

        this.elements.fileName.textContent = file.name;
        this.elements.fileSize.textContent = this.formatFileSize(file.size);
    }

    static resetFileSelection() {
        if (!this.elements.fileInput) return;
        this.elements.fileInput.value = "";
    }

    static isValidVideoFile(file) {
        const validTypes = [
            "video/mp4",
            "video/webm", 
            "video/quicktime",
            "video/avi"
        ];
        return validTypes.includes(file.type);
    }

    static formatFileSize(bytes) {
        if (bytes === 0) return "0 Bytes";
        const k = 1024;
        const sizes = ["Bytes", "KB", "MB", "GB"];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
    }

    // ============================================================================
    // МЕТОДЫ ДЛЯ DRAG & DROP ЭФФЕКТОВ
    // ============================================================================

    static addDragEffects() {
        if (!this.elements.fileDropZone) return;

        this.elements.fileDropZone.classList.add(
            "border-indigo-600",
            "bg-indigo-100", 
            "scale-105",
            "shadow-lg"
        );
        this.elements.fileDropZone.classList.remove(
            "border-indigo-400",
            "bg-indigo-50"
        );
    }

    static removeDragEffects() {
        if (!this.elements.fileDropZone) return;

        this.elements.fileDropZone.classList.remove(
            "border-indigo-600",
            "bg-indigo-100",
            "scale-105", 
            "shadow-lg"
        );

        // Возвращаем правильный стиль в зависимости от состояния
        if (!appState.file) {
            this.elements.fileDropZone.classList.add("border-indigo-400", "bg-indigo-50");
        } else {
            this.elements.fileDropZone.classList.add("border-green-400", "bg-green-50");
        }
    }

    // ============================================================================
    // МЕТОДЫ ДЛЯ ОТОБРАЖЕНИЯ ПРОГРЕССА И РЕЗУЛЬТАТОВ
    // ============================================================================

    static updateProgress(stage, progress, message) {
        if (!this.elements.progressStage || !this.elements.progressPercentage || 
            !this.elements.progressBar || !this.elements.progressMessage) return;

        const stageNames = {
            initializing: "Инициализация",
            loading: "Загрузка видео",
            processing: "Анализ кадров",
            detection: "Детекция нарушений",
            finalizing: "Формирование результатов",
            completed: "Завершено"
        };

        this.elements.progressStage.textContent = stageNames[stage] || stage;
        this.elements.progressPercentage.textContent = `${progress}%`;
        this.elements.progressBar.style.width = `${progress}%`;
        this.elements.progressMessage.textContent = message;

        // Обновляем цвет прогресс-бара
        this.elements.progressBar.className = this.getProgressBarClass(progress, stage);
    }

    static getProgressBarClass(progress, stage) {
        const baseClass = "h-2.5 rounded-full transition-all duration-300";
        
        if (progress === 100) {
            return `bg-green-600 ${baseClass}`;
        } else if (stage === "error") {
            return `bg-red-600 ${baseClass}`;
        } else {
            return `bg-blue-600 ${baseClass}`;
        }
    }

    static showResults(results) {
        if (!this.elements.resultsDiv || !this.elements.containerDiv) return;

        const { images } = results || { images: [] };
        
        // Расширяем контейнер если есть изображения
        if (images.length > 0) {
            this.elements.containerDiv.classList.remove("max-w-4xl");
            this.elements.containerDiv.classList.add("max-w-6xl");
        }

        this.elements.resultsDiv.innerHTML = this.generateResultsHTML(images);
    }

    static generateResultsHTML(images) {
        return `
            <div class="mt-4">
                <h3 class="text-lg font-semibold text-gray-800 mb-4">Результаты обработки:</h3>
                <div class="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                    ${images.map(img => this.generateImageHTML(img)).join("")}
                </div>
                <div class="mt-6 text-center">
                    <button 
                        onclick="UI.loadNewVideo()" 
                        class="inline-flex items-center px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white font-medium rounded-lg transition-colors duration-200"
                    >
                        <svg class="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6v6m0 0v6m0-6h6m-6 0H6"></path>
                        </svg>
                        Загрузить новое видео
                    </button>
                </div>
            </div>
        `;
    }

    static generateImageHTML(img) {
        return `
            <div class="relative group">
                <img src="data:image/jpeg;base64,${img}" 
                     class="w-full h-auto rounded-lg shadow-lg transition-transform transform group-hover:scale-105 cursor-pointer" 
                     alt="Изображение нарушения" 
                     style="max-height: 300px; object-fit: cover;" 
                     onclick="UI.openFullScreen('${img}')">
                <div class="absolute inset-0 bg-black bg-opacity-40 flex items-center justify-center rounded-lg opacity-0 group-hover:opacity-100 transition-opacity">
                    <span class="text-white font-semibold text-lg">Нарушение</span>
                </div>
            </div>
        `;
    }

    static showError(error) {
        if (!this.elements.resultsDiv) return;

        this.elements.resultsDiv.innerHTML = `
            <div class="mt-4 p-4 bg-red-100 border border-red-400 text-red-700 rounded-lg">
                <strong>Ошибка обработки:</strong> ${error}
            </div>
            <div class="mt-4 text-center">
                <button 
                    onclick="UI.loadNewVideo()" 
                    class="inline-flex items-center px-4 py-2 bg-red-600 hover:bg-red-700 text-white font-medium rounded-lg transition-colors duration-200"
                >
                    <svg class="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
                    </svg>
                    Попробовать снова
                </button>
            </div>
        `;
    }

    static openFullScreen(image) {
        const fullScreenDiv = document.createElement("div");
        Object.assign(fullScreenDiv.style, {
            position: "fixed",
            top: 0,
            left: 0,
            width: "100%",
            height: "100%",
            backgroundColor: "rgba(0, 0, 0, 0.8)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 1000
        });

        const img = document.createElement("img");
        img.src = `data:image/jpeg;base64,${image}`;
        Object.assign(img.style, {
            maxWidth: "90%",
            maxHeight: "90%",
            borderRadius: "8px"
        });

        fullScreenDiv.appendChild(img);
        fullScreenDiv.onclick = () => document.body.removeChild(fullScreenDiv);
        document.body.appendChild(fullScreenDiv);
    }

    static loadNewVideo() {
        // Очищаем результаты
        if (this.elements.resultsDiv) {
            this.elements.resultsDiv.innerHTML = "";
        }

        // Восстанавливаем размер контейнера
        if (this.elements.containerDiv) {
            this.elements.containerDiv.classList.remove("max-w-6xl");
            this.elements.containerDiv.classList.add("max-w-4xl");
        }

        // Скрываем прогресс
        if (this.elements.progressContainer) {
            this.elements.progressContainer.style.display = "none";
        }

        // Сбрасываем состояние
        appState.clearFile();
        appState.setUploading(false);
        appState.setProcessing(false);
        this.resetFileSelection();

        // Явно показываем дроп зону
        appState.showDropZone();

        // Прокручиваем к форме загрузки
        if (this.elements.uploadForm) {
            this.elements.uploadForm.scrollIntoView({ behavior: "smooth" });
        }
    }
}

// ============================================================================
// ИНИЦИАЛИЗАЦИЯ ПРИЛОЖЕНИЯ
// ============================================================================

// Глобальный экземпляр состояния приложения
const appState = new AppState();

// Инициализация при загрузке страницы
document.addEventListener('DOMContentLoaded', function() {
    UI.initialize();
});

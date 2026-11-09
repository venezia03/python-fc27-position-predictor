const API = "http://127.0.0.1:8000";

const groups = {
    "Основные характеристики": [
        "pace", "shooting", "passing", "dribbling",
        "defending", "physicality"
    ],
    "Техника": [
        "skill_moves", "weak_foot", "attacking_crossing",
        "attacking_finishing", "attacking_heading_accuracy",
        "attacking_short_passing", "attacking_volleys",
        "skill_dribbling", "skill_curve", "skill_fk_accuracy",
        "skill_long_passing", "skill_ball_control"
    ],
    "Скорость и движение": [
        "movement_acceleration", "movement_sprint_speed",
        "movement_agility", "movement_reactions",
        "movement_balance"
    ],
    "Физические показатели": [
        "power_shot_power", "power_jumping", "power_stamina",
        "power_strength", "power_long_shots"
    ],
    "Игровое мышление": [
        "mentality_aggression", "mentality_interceptions",
        "mentality_positioning", "mentality_vision",
        "mentality_penalties", "mentality_composure"
    ],
    "Защита и вратарские навыки": [
        "defending_awareness", "defending_standing_tackle",
        "defending_sliding_tackle", "goalkeeping_diving",
        "goalkeeping_handling", "goalkeeping_kicking",
        "goalkeeping_positioning", "goalkeeping_reflexes"
    ]
};

const root = document.getElementById("fields");
const form = document.getElementById("playerForm");
const result = document.getElementById("result");

function readableName(name) {
    return name
        .replaceAll("_", " ")
        .replace(/\b\w/g, c => c.toUpperCase());
}

// Создание групп характеристик
Object.entries(groups).forEach(([groupName, fields]) => {
    const section = document.createElement("div");
    section.className = "field-group";

    const heading = document.createElement("h3");
    heading.textContent = groupName;
    section.appendChild(heading);

    fields.forEach(name => {
        const label = document.createElement("label");
        label.textContent = readableName(name);

        const input = document.createElement("input");
        input.type = "number";
        input.min = 0;
        input.max = 99;
        input.value = 50;
        input.name = name;
        input.required = true;

        label.appendChild(input);
        section.appendChild(label);
    });

    root.appendChild(section);
});

// Предпочитаемая нога
const footSection = document.createElement("div");
footSection.className = "field-group";

const footHeading = document.createElement("h3");
footHeading.textContent = "Дополнительные данные";
footSection.appendChild(footHeading);

const footLabel = document.createElement("label");
footLabel.textContent = "Preferred Foot";

const footSelect = document.createElement("select");
footSelect.name = "preferred_foot";

["Right", "Left"].forEach(value => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value === "Right" ? "Правая" : "Левая";
    footSelect.appendChild(option);
});

footLabel.appendChild(footSelect);
footSection.appendChild(footLabel);
root.appendChild(footSection);

// Отображение результата предсказания
function showPrediction(data) {
    const prediction = data.predictions[0];
    const probabilities = Object.entries(prediction.probabilities)
        .sort((a, b) => b[1] - a[1]);

    result.classList.remove("hidden");

    result.innerHTML = `
        <div class="eyebrow">ANALYSIS COMPLETE</div>
        <h2 style="margin:12px 0;color:var(--accent)">
            ${prediction.predicted_position}
        </h2>
        <p style="color:var(--muted)">Предсказанная игровая позиция</p>
        <h3>Вероятности классов</h3>
        <div id="probabilityList"></div>
    `;

    const list = result.querySelector("#probabilityList");

    probabilities.forEach(([position, probability]) => {
        const percent = (probability * 100).toFixed(2);

        const item = document.createElement("div");
        item.style.margin = "14px 0";

        const label = document.createElement("div");
        label.style.display = "flex";
        label.style.justifyContent = "space-between";
        label.style.marginBottom = "6px";
        label.innerHTML = `<strong>${position}</strong><span>${percent}%</span>`;

        const track = document.createElement("div");
        track.style.height = "7px";
        track.style.background = "#263348";
        track.style.borderRadius = "10px";
        track.style.overflow = "hidden";

        const bar = document.createElement("div");
        bar.style.width = `${Math.min(Number(percent), 100)}%`;
        bar.style.height = "100%";
        bar.style.background = "var(--accent)";
        bar.style.borderRadius = "10px";

        track.appendChild(bar);
        item.append(label, track);
        list.appendChild(item);
    });

    result.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// Отправка игрока на API
form.addEventListener("submit", async event => {
    event.preventDefault();

    const button = form.querySelector('button[type="submit"]');
    const originalText = button.innerHTML;

    const player = {};

    new FormData(form).forEach((value, key) => {
        player[key] = key === "preferred_foot" ? value : Number(value);
    });

    button.disabled = true;
    button.textContent = "Анализируем...";
    result.classList.remove("hidden");
    result.textContent = "Модель обрабатывает характеристики игрока...";

    try {
        const response = await fetch(`${API}/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(player)
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Ошибка API");
        }

        showPrediction(data);

    } catch (error) {
        result.classList.remove("hidden");
        result.textContent = `Ошибка: ${error.message}`;
    } finally {
        button.disabled = false;
        button.innerHTML = originalText;
    }
});

// Отображение имени выбранного CSV
const csvInput = document.getElementById("csv");

csvInput.addEventListener("change", () => {
    document.getElementById("fileName").textContent =
        csvInput.files[0]
            ? csvInput.files[0].name
            : "CSV · MAX SIZE DEPENDS ON SERVER";
});

// Загрузка и анализ CSV
document.getElementById("upload").addEventListener("click", async () => {
    const file = csvInput.files[0];
    const output = document.getElementById("csvResult");
    const button = document.getElementById("upload");

    if (!file) {
        output.classList.remove("hidden");
        output.textContent = "Сначала выбери CSV-файл.";
        return;
    }

    if (!file.name.toLowerCase().endsWith(".csv")) {
        output.classList.remove("hidden");
        output.textContent = "Поддерживаются только CSV-файлы.";
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    button.disabled = true;
    button.textContent = "Обрабатываем CSV...";
    output.classList.remove("hidden");
    output.textContent = "Идёт обработка данных...";

    try {
        const response = await fetch(`${API}/predict-from-csv`, {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Ошибка API");
        }

        output.innerHTML = `
            <div class="eyebrow">CSV ANALYSIS COMPLETE</div>
            <h2 style="color:var(--accent)">${data.count.toLocaleString("ru-RU")}</h2>
            <p>Строк обработано успешно</p>
            <p>Предсказания добавлены в результат обработки.</p>
            <p style="color:var(--muted);font-size:12px">
                Для полного просмотра результатов используйте API.
            </p>
        `;

    } catch (error) {
        output.textContent = `Ошибка: ${error.message}`;
    } finally {
        button.disabled = false;
        button.innerHTML = 'Запустить анализ <span>→</span>';
    }
});
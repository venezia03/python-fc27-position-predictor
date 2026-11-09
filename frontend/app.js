
const API = "http://127.0.0.1:8000";

const fields = [
    { name: "pace", label: "PAC — Скорость" },
    { name: "shooting", label: "SHO — Удар" },
    { name: "passing", label: "PAS — Пас" },
    { name: "dribbling", label: "DRI — Дриблинг" },
    { name: "defending", label: "DEF — Защита" },
    { name: "physicality", label: "PHY — Физика" }
];

const root = document.getElementById("fields");
const form = document.getElementById("playerForm");
const result = document.getElementById("result");

fields.forEach(field => {
    const label = document.createElement("label");
    label.textContent = field.label;

    const input = document.createElement("input");
    input.type = "number";
    input.min = 0;
    input.max = 99;
    input.value = 50;
    input.name = field.name;
    input.required = true;

    label.appendChild(input);
    root.appendChild(label);
});

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
        <h3>Вероятности позиций</h3>
        <div id="probabilityList"></div>
    `;

    const list = result.querySelector("#probabilityList");

    probabilities.forEach(([position, probability]) => {
        const percent = (probability * 100).toFixed(2);

        const item = document.createElement("div");
        item.style.margin = "14px 0";

        item.innerHTML = `
            <div style="display:flex;justify-content:space-between;margin-bottom:6px">
                <strong>${position}</strong>
                <span>${percent}%</span>
            </div>
            <div style="height:7px;background:#263348;border-radius:10px;overflow:hidden">
                <div style="width:${percent}%;height:100%;background:var(--accent)"></div>
            </div>
        `;

        list.appendChild(item);
    });

    result.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

form.addEventListener("submit", async event => {
    event.preventDefault();

    const button = form.querySelector('button[type="submit"]');
    const originalText = button.innerHTML;
    const player = {};

    fields.forEach(field => {
        player[field.name] = Number(
            form.elements[field.name].value
        );
    });

    if (Object.values(player).some(
        value => !Number.isFinite(value) || value < 0 || value > 99
    )) {
        result.classList.remove("hidden");
        result.textContent = "Введи значения от 0 до 99.";
        return;
    }

    button.disabled = true;
    button.textContent = "Анализируем...";
    result.classList.remove("hidden");
    result.textContent = "Модель анализирует характеристики...";

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

const csvInput = document.getElementById("csv");

csvInput.addEventListener("change", () => {
    document.getElementById("fileName").textContent =
        csvInput.files[0]?.name || "CSV · MAX SIZE DEPENDS ON SERVER";
});

document.getElementById("upload").addEventListener("click", async () => {
    const file = csvInput.files[0];
    const output = document.getElementById("csvResult");
    const button = document.getElementById("upload");

    if (!file || !file.name.toLowerCase().endsWith(".csv")) {
        output.classList.remove("hidden");
        output.textContent = "Выбери CSV-файл.";
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    button.disabled = true;
    button.textContent = "Обрабатываем...";
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
            <h2 style="color:var(--accent)">${data.count}</h2>
            <p>Игроков обработано</p>
        `;

    } catch (error) {
        output.textContent = `Ошибка: ${error.message}`;
    } finally {
        button.disabled = false;
        button.innerHTML = 'Запустить анализ <span>→</span>';
    }
});

document.addEventListener("DOMContentLoaded", () => {

    const imageInput = document.getElementById("imageInput");
    const dropzone = document.getElementById("dropzone");
    const preview = document.getElementById("preview");

    const analyzeBtn = document.getElementById("analyzeBtn");
    const downloadReportBtn = document.getElementById("downloadReportBtn");

    const loadingIndicator = document.getElementById("loadingIndicator");
    const demoMessage = document.getElementById("demoMessage");

    const comparisonBox = document.getElementById("comparisonBox");
    const modelResults = document.getElementById("modelResults");

    const visualizationSection = document.getElementById("visualizationSection");
    const visualResults = document.getElementById("visualResults");

    let selectedFile = null;
    let latestReport = null;


    // --------------------------------------------------------
    // IMAGE SELECTION
    // --------------------------------------------------------

    dropzone.addEventListener("click", () => {
        imageInput.click();
    });

    imageInput.addEventListener("change", event => {

        const file = event.target.files[0];

        if (file) {
            selectImage(file);
        }

    });


    function selectImage(file) {

        const allowedTypes = [
            "image/jpeg",
            "image/png",
            "image/webp"
        ];

        if (!allowedTypes.includes(file.type)) {
            showMessage("Please select JPG, PNG or WEBP.", true);
            return;
        }

        if (file.size > 10 * 1024 * 1024) {
            showMessage("Image must be smaller than 10 MB.", true);
            return;
        }

        selectedFile = file;

        preview.src = URL.createObjectURL(file);
        preview.style.display = "block";

        analyzeBtn.disabled = false;

        comparisonBox.hidden = true;
        visualizationSection.hidden = true;

        downloadReportBtn.hidden = true;

        latestReport = null;

        showMessage("Image selected successfully.");

    }


    // --------------------------------------------------------
    // DRAG AND DROP
    // --------------------------------------------------------

    dropzone.addEventListener("dragover", event => {

        event.preventDefault();

        dropzone.classList.add("dragover");

    });

    dropzone.addEventListener("dragleave", () => {

        dropzone.classList.remove("dragover");

    });

    dropzone.addEventListener("drop", event => {

        event.preventDefault();

        dropzone.classList.remove("dragover");

        const file = event.dataTransfer.files[0];

        if (file) {
            selectImage(file);
        }

    });


    // --------------------------------------------------------
    // MESSAGES
    // --------------------------------------------------------

    function showMessage(message, isError = false) {

        demoMessage.textContent = message;

        demoMessage.style.color = isError ? "#dc3545" : "inherit";

    }


    // --------------------------------------------------------
    // ANALYZE BUTTON
    // --------------------------------------------------------

    analyzeBtn.addEventListener("click", analyzeImage);


    async function analyzeImage() {

        if (!selectedFile) {
            showMessage("Please upload an image first.", true);
            return;
        }

        analyzeBtn.disabled = true;

        loadingIndicator.style.display = "block";

        comparisonBox.hidden = true;
        visualizationSection.hidden = true;

        downloadReportBtn.hidden = true;

        modelResults.innerHTML = "";
        visualResults.innerHTML = "";

        showMessage("Running predictions using three models...");

        try {

            const formData = new FormData();

            formData.append("image", selectedFile);

            const response = await fetch("/api/compare", {
                method: "POST",
                body: formData
            });

            const data = await response.json();

            if (!response.ok || !data.success) {
                throw new Error(data.error || "Prediction failed.");
            }

            // Save prediction results.
            latestReport = {
                filename: data.filename,
                results: data.results,
                comparison: data.comparison,
                gradcam: []
            };

            // DISPLAY PREDICTIONS IMMEDIATELY.
            displayPredictions(data);

            comparisonBox.hidden = false;

            loadingIndicator.style.display = "none";

            showMessage("Predictions completed. Generating Grad-CAM images separately...");

            // Start Grad-CAM only after predictions are visible.
            await generateAllGradCAM(data.session_id);

        } catch (error) {

            console.error("Analysis error:", error);

            showMessage(error.message || "Something went wrong.", true);

            loadingIndicator.style.display = "none";

        } finally {

            analyzeBtn.disabled = false;

        }

    }


    // --------------------------------------------------------
    // DISPLAY PREDICTIONS
    // --------------------------------------------------------

    function displayPredictions(data) {

        modelResults.innerHTML = "";

        data.results.forEach(result => {

            const card = document.createElement("div");

            card.className = "result-card";

            const predictionColor =
                result.prediction === "Cancer" ? "#dc3545" : "#198754";

            card.innerHTML = `
                <h3>${escapeHTML(result.model)}</h3>

                <div class="prediction-label"
                     style="color:${predictionColor}">
                    ${escapeHTML(result.prediction)}
                </div>

                <p>Confidence: ${result.confidence}%</p>

                <p>Cancer probability:
                   ${result.cancer_probability}%</p>

                <div class="probability-track">
                    <div class="probability-fill"
                         style="width:${result.cancer_probability}%">
                    </div>
                </div>

                <p>Non-cancer probability:
                   ${result.non_cancer_probability}%</p>
            `;

            modelResults.appendChild(card);

        });


        const comparison = data.comparison;

        document.getElementById("majorityPrediction").textContent =
            comparison.majority_prediction;

        document.getElementById("modelAgreement").textContent =
            comparison.agreement;

        document.getElementById("averageProbability").textContent =
            comparison.average_cancer_probability + "%";

        document.getElementById("probabilityRange").textContent =
            comparison.probability_range + " percentage points";

        document.getElementById("cancerVotes").textContent =
            comparison.cancer_votes;

        document.getElementById("nonCancerVotes").textContent =
            comparison.non_cancer_votes;

    }


    // --------------------------------------------------------
    // GRAD-CAM GENERATION
    // ONE MODEL AT A TIME
    // --------------------------------------------------------

    async function generateAllGradCAM(sessionId) {

        const models = [
            "MobileNetV2",
            "EfficientNetB0",
            "ResNet50"
        ];

        visualizationSection.hidden = false;

        visualResults.innerHTML = "";

        const originalCard = document.createElement("div");

        originalCard.className = "visual-card";

        originalCard.innerHTML = `
            <h3>Original Image</h3>
            <img src="${preview.src}" alt="Original uploaded image">
            <p>Original uploaded image.</p>
        `;

        visualResults.appendChild(originalCard);


        for (let i = 0; i < models.length; i++) {

            const modelName = models[i];

            showMessage(
                `Generating Grad-CAM ${i + 1} of 3: ${modelName}. Predictions are already available.`
            );

            const card = document.createElement("div");

            card.className = "visual-card";

            card.innerHTML = `
                <h3>${modelName}</h3>
                <p id="gradcam-status-${i}">Generating visualization...</p>
            `;

            visualResults.appendChild(card);

            try {

                const response = await fetch("/api/gradcam", {

                    method: "POST",

                    headers: {
                        "Content-Type": "application/json"
                    },

                    body: JSON.stringify({
                        session_id: sessionId,
                        model: modelName
                    })

                });

                const data = await response.json();

                if (!response.ok || !data.success) {
                    throw new Error(data.error || "Grad-CAM failed.");
                }

                const result = data.result;

                card.innerHTML = `
                    <h3>${modelName}</h3>

                    <img
                        src="${result.image_url}?t=${Date.now()}"
                        alt="${modelName} Grad-CAM overlay"
                    >

                    <p>Experimental feature-attribution visualization.</p>
                `;

                latestReport.gradcam.push(result);

            } catch (error) {

                console.error(modelName + " Grad-CAM error:", error);

                card.innerHTML = `
                    <h3>${modelName}</h3>
                    <p style="color:#dc3545">
                        Grad-CAM unavailable.
                    </p>
                    <p>${escapeHTML(error.message)}</p>
                `;

            }

        }


        showMessage(
            "Analysis completed. Predictions and Grad-CAM processing finished."
        );

        downloadReportBtn.hidden = false;

    }


    // --------------------------------------------------------
    // DOWNLOAD REPORT
    // --------------------------------------------------------

    downloadReportBtn.addEventListener("click", () => {

        if (!latestReport) {
            showMessage("No report is available.", true);
            return;
        }

        const reportWindow = window.open("", "_blank");

        if (!reportWindow) {
            showMessage("Please allow pop-ups to download the report.", true);
            return;
        }

        const rows = latestReport.results.map(result => `
            <tr>
                <td>${escapeHTML(result.model)}</td>
                <td>${escapeHTML(result.prediction)}</td>
                <td>${result.cancer_probability}%</td>
                <td>${result.non_cancer_probability}%</td>
                <td>${result.confidence}%</td>
            </tr>
        `).join("");


        const gradcamRows = latestReport.gradcam.map(item => `
            <h3>${escapeHTML(item.model)}</h3>
            <img src="${item.image_url}" style="width:350px;max-width:100%">
        `).join("");


        reportWindow.document.write(`
            <!DOCTYPE html>
            <html>
            <head>
                <title>OncoLens Research Report</title>

                <style>
                    body {
                        font-family: Arial, sans-serif;
                        padding: 30px;
                    }

                    table {
                        border-collapse: collapse;
                        width: 100%;
                    }

                    th, td {
                        border: 1px solid #ccc;
                        padding: 10px;
                        text-align: left;
                    }

                    img {
                        margin-bottom: 20px;
                    }

                    @media print {
                        button {
                            display: none;
                        }
                    }
                </style>
            </head>

            <body>

                <h1>OncoLens Model Comparison Report</h1>

                <p>Image: ${escapeHTML(latestReport.filename)}</p>

                <h2>Model Predictions</h2>

                <table>
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>Prediction</th>
                            <th>Cancer Probability</th>
                            <th>Non-Cancer Probability</th>
                            <th>Confidence</th>
                        </tr>
                    </thead>

                    <tbody>
                        ${rows}
                    </tbody>
                </table>

                <h2>Overall Comparison</h2>

                <p>Majority prediction:
                    ${escapeHTML(latestReport.comparison.majority_prediction)}
                </p>

                <p>Model agreement:
                    ${escapeHTML(latestReport.comparison.agreement)}
                </p>

                <h2>Grad-CAM Visualizations</h2>

                ${gradcamRows}

                <hr>

                <p>
                    Research prototype only. Not clinically validated
                    and not intended for medical diagnosis.
                </p>

                <button onclick="window.print()">Print / Save as PDF</button>

            </body>
            </html>
        `);

        reportWindow.document.close();

    });


    // --------------------------------------------------------
    // HTML ESCAPING
    // --------------------------------------------------------

    function escapeHTML(value) {

        return String(value).replace(/[&<>"']/g, character => {

            const entities = {
                "&": "&amp;",
                "<": "&lt;",
                ">": "&gt;",
                '"': "&quot;",
                "'": "&#039;"
            };

            return entities[character];

        });

    }

});
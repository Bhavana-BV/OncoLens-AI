document.addEventListener("DOMContentLoaded", function () {

    // ==================================================
    // ELEMENTS
    // ==================================================

    const imageInput = document.getElementById("imageInput");
    const preview = document.getElementById("preview");

    const dropzone = document.getElementById("dropzone");

    const analyzeBtn = document.getElementById("analyzeBtn");
    const downloadReportBtn =
        document.getElementById("downloadReportBtn");

    const demoMessage = document.getElementById("demoMessage");
    const loadingIndicator =
        document.getElementById("loadingIndicator");

    const comparisonBox =
        document.getElementById("comparisonBox");

    const modelResults =
        document.getElementById("modelResults");

    const visualizationSection =
        document.getElementById("visualizationSection");

    const visualResults =
        document.getElementById("visualResults");


    // ==================================================
    // STATE
    // ==================================================

    let selectedFile = null;
    let latestData = null;
    let originalImageData = null;


    // ==================================================
    // CONFIGURATION
    // ==================================================

    const MAX_FILE_SIZE = 10 * 1024 * 1024;

    const ALLOWED_TYPES = [
        "image/jpeg",
        "image/png",
        "image/webp"
    ];


    // ==================================================
    // HELPERS
    // ==================================================

    function setMessage(message, error = false) {

        demoMessage.textContent = message;

        demoMessage.style.color = error
            ? "#dc3545"
            : "inherit";
    }


    function makeElement(tag, className, text) {

        const element = document.createElement(tag);

        if (className) {
            element.className = className;
        }

        if (text !== undefined) {
            element.textContent = text;
        }

        return element;
    }


    function formatPercent(value) {

        const number = Number(value);

        if (!Number.isFinite(number)) {
            return "N/A";
        }

        return number.toFixed(2) + "%";
    }


    function clearResults() {

        latestData = null;

        comparisonBox.hidden = true;
        visualizationSection.hidden = true;

        downloadReportBtn.hidden = true;

        modelResults.replaceChildren();
        visualResults.replaceChildren();

        const ids = [
            "majorityPrediction",
            "modelAgreement",
            "averageProbability",
            "probabilityRange",
            "cancerVotes",
            "nonCancerVotes"
        ];

        ids.forEach(function (id) {

            document.getElementById(id).textContent = "—";

        });

    }


    // ==================================================
    // VALIDATION
    // ==================================================

    function validateFile(file) {

        if (!file) {
            return "Please select an image.";
        }

        if (!ALLOWED_TYPES.includes(file.type)) {
            return "Please upload a JPG, PNG or WEBP image.";
        }

        if (file.size > MAX_FILE_SIZE) {
            return "Image must be smaller than 10 MB.";
        }

        if (file.size === 0) {
            return "The selected image is empty.";
        }

        return null;

    }


    // ==================================================
    // IMAGE SELECTION
    // ==================================================

    function handleFile(file) {

        const error = validateFile(file);

        if (error) {

            selectedFile = null;
            originalImageData = null;

            preview.removeAttribute("src");
            preview.style.display = "none";

            analyzeBtn.disabled = true;

            clearResults();

            setMessage(error, true);

            return;

        }

        selectedFile = file;

        const reader = new FileReader();

        reader.onload = function (event) {

            originalImageData = event.target.result;

            preview.src = originalImageData;
            preview.style.display = "block";

        };

        reader.onerror = function () {

            setMessage(
                "Unable to read the selected image.",
                true
            );

        };

        reader.readAsDataURL(file);

        analyzeBtn.disabled = false;

        clearResults();

        setMessage(
            "Image selected. Click Analyze Image to continue."
        );

    }


    imageInput.addEventListener("change", function () {

        handleFile(imageInput.files[0]);

    });


    // ==================================================
    // DRAG AND DROP
    // ==================================================

    dropzone.addEventListener("click", function () {

        imageInput.click();

    });


    ["dragenter", "dragover"].forEach(function (eventName) {

        dropzone.addEventListener(eventName, function (event) {

            event.preventDefault();

            dropzone.classList.add("dragover");

        });

    });


    ["dragleave", "drop"].forEach(function (eventName) {

        dropzone.addEventListener(eventName, function (event) {

            event.preventDefault();

            dropzone.classList.remove("dragover");

        });

    });


    dropzone.addEventListener("drop", function (event) {

        const files = event.dataTransfer.files;

        if (files.length > 0) {

            handleFile(files[0]);

        }

    });


    // ==================================================
    // MODEL RESULT CARDS
    // ==================================================

    function displayModelResults(results) {

        modelResults.replaceChildren();

        results.forEach(function (result) {

            const card = makeElement(
                "div",
                "result-card"
            );

            const title = makeElement(
                "h3",
                "",
                result.model
            );

            const prediction = makeElement(
                "div",
                "prediction-label",
                result.prediction
            );

            const confidence = makeElement(
                "p",
                "",
                "Confidence: " +
                formatPercent(result.confidence)
            );

            const probability = makeElement(
                "p",
                "",
                "Cancer probability: " +
                formatPercent(result.cancer_probability)
            );

            const bar = makeElement(
                "div",
                "probability-track"
            );

            const fill = makeElement(
                "div",
                "probability-fill"
            );

            fill.style.width =
                Math.min(
                    100,
                    Math.max(
                        0,
                        Number(result.cancer_probability) || 0
                    )
                ) + "%";

            bar.appendChild(fill);

            card.appendChild(title);
            card.appendChild(prediction);
            card.appendChild(confidence);
            card.appendChild(probability);
            card.appendChild(bar);

            modelResults.appendChild(card);

        });

    }


    // ==================================================
    // COMPARISON SUMMARY
    // ==================================================

    function displayComparison(comparison) {

        if (!comparison) {
            return;
        }

        document.getElementById(
            "majorityPrediction"
        ).textContent =
            comparison.majority_prediction;

        document.getElementById(
            "modelAgreement"
        ).textContent =
            comparison.agreement;

        document.getElementById(
            "averageProbability"
        ).textContent =
            formatPercent(
                comparison.average_cancer_probability
            );

        document.getElementById(
            "probabilityRange"
        ).textContent =
            formatPercent(
                comparison.probability_range
            ) + " points";

        document.getElementById(
            "cancerVotes"
        ).textContent =
            comparison.cancer_votes;

        document.getElementById(
            "nonCancerVotes"
        ).textContent =
            comparison.non_cancer_votes;

    }


    // ==================================================
    // ORIGINAL IMAGE + GRAD-CAM
    // ==================================================

    function displayVisualizations(results) {

        visualResults.replaceChildren();

        // Original uploaded image

        const originalCard = makeElement(
            "div",
            "visual-card"
        );

        originalCard.appendChild(
            makeElement(
                "h3",
                "",
                "Original Uploaded Image"
            )
        );

        const originalImage = document.createElement("img");

        originalImage.src = originalImageData;

        originalImage.alt = "Original uploaded image";

        originalCard.appendChild(originalImage);

        originalCard.appendChild(
            makeElement(
                "p",
                "",
                "Original image supplied for model analysis."
            )
        );

        visualResults.appendChild(originalCard);


        // Grad-CAM images

        results.forEach(function (result) {

            const card = makeElement(
                "div",
                "visual-card"
            );

            card.appendChild(
                makeElement(
                    "h3",
                    "",
                    result.model + " Grad-CAM"
                )
            );

            if (result.image_url) {

                const image = document.createElement("img");

                image.src =
                    result.image_url +
                    "?v=" + Date.now();

                image.alt =
                    "Grad-CAM overlay for " +
                    result.model;

                image.loading = "lazy";

                image.onerror = function () {

                    image.style.display = "none";

                    card.appendChild(
                        makeElement(
                            "p",
                            "",
                            "Unable to load this visualization."
                        )
                    );

                };

                card.appendChild(image);

            } else {

                card.appendChild(
                    makeElement(
                        "p",
                        "",
                        "Grad-CAM unavailable for this model."
                    )
                );

            }

            card.appendChild(
                makeElement(
                    "p",
                    "",
                    result.gradcam_note ||
                    "Experimental feature-attribution visualization."
                )
            );

            visualResults.appendChild(card);

        });

        visualizationSection.hidden = false;

    }


    // ==================================================
    // ANALYSIS
    // ==================================================

    analyzeBtn.addEventListener("click", async function () {

        if (!selectedFile) {

            setMessage(
                "Please select an image first.",
                true
            );

            return;

        }

        const error = validateFile(selectedFile);

        if (error) {

            setMessage(error, true);

            return;

        }

        analyzeBtn.disabled = true;

        loadingIndicator.style.display = "block";

        setMessage(
            "Processing image. Please wait..."
        );

        clearResults();

        const formData = new FormData();

        formData.append(
            "image",
            selectedFile
        );

        try {

            const response = await fetch(
                "/api/compare",
                {
                    method: "POST",
                    body: formData
                }
            );

            const data = await response.json();

            if (!response.ok) {

                throw new Error(
                    data.error || "Analysis failed."
                );

            }

            latestData = data;

            displayModelResults(
                data.results || []
            );

            displayComparison(
                data.comparison
            );

            displayVisualizations(
                data.gradcam_results || []
            );

            comparisonBox.hidden = false;

            downloadReportBtn.hidden = false;

            setMessage(
                "Analysis completed successfully."
            );

        } catch (error) {

            console.error(error);

            setMessage(
                error.message ||
                "Unable to process the image.",
                true
            );

        } finally {

            analyzeBtn.disabled = false;

            loadingIndicator.style.display = "none";

        }

    });


    // ==================================================
    // DOWNLOAD REPORT
    // ==================================================

    async function imageToDataURL(url) {

        try {

            const response = await fetch(url);

            if (!response.ok) {
                return null;
            }

            const blob = await response.blob();

            return await new Promise(function (resolve) {

                const reader = new FileReader();

                reader.onloadend = function () {
                    resolve(reader.result);
                };

                reader.onerror = function () {
                    resolve(null);
                };

                reader.readAsDataURL(blob);

            });

        } catch (error) {

            console.error(
                "Report image error:",
                error
            );

            return null;

        }

    }


    downloadReportBtn.addEventListener(
        "click",
        async function () {

            if (!latestData) {

                setMessage(
                    "Please analyze an image first.",
                    true
                );

                return;

            }

            downloadReportBtn.disabled = true;

            downloadReportBtn.textContent =
                "Preparing Report...";

            try {

                const results = latestData.results || [];

                const comparison = latestData.comparison || {};

                const gradcamResults =
                    latestData.gradcam_results || [];


                const originalImage =
                    originalImageData;


                const heatmapImages = [];

                for (const item of gradcamResults) {

                    let imageData = null;

                    if (item.image_url) {

                        imageData = await imageToDataURL(
                            item.image_url
                        );

                    }

                    heatmapImages.push({
                        model: item.model,
                        image: imageData
                    });

                }


                const reportWindow = window.open(
                    "",
                    "_blank"
                );

                if (!reportWindow) {

                    throw new Error(
                        "Please allow pop-ups to generate the report."
                    );

                }


                let modelRows = "";

                results.forEach(function (result) {

                    modelRows += `
                        <tr>
                            <td>${escapeHTML(result.model)}</td>
                            <td>${escapeHTML(result.prediction)}</td>
                            <td>${result.confidence}%</td>
                            <td>${result.cancer_probability}%</td>
                        </tr>
                    `;

                });


                let heatmapSections = "";

                heatmapImages.forEach(function (item) {

                    heatmapSections += `
                        <section>
                            <h3>${escapeHTML(item.model)} Grad-CAM</h3>

                            ${
                                item.image
                                ? `<img src="${item.image}" alt="Grad-CAM">`
                                : `<p>Image unavailable.</p>`
                            }

                        </section>
                    `;

                });


                const reportHTML = `
                    <!DOCTYPE html>
                    <html>
                    <head>

                        <meta charset="UTF-8">

                        <title>OncoLens Prediction Report</title>

                        <style>

                            body {
                                font-family: Arial, sans-serif;
                                max-width: 900px;
                                margin: 35px auto;
                                padding: 20px;
                                color: #222;
                            }

                            h1, h2 {
                                color: #294f9e;
                            }

                            table {
                                width: 100%;
                                border-collapse: collapse;
                                margin: 20px 0;
                            }

                            th, td {
                                border: 1px solid #ddd;
                                padding: 12px;
                                text-align: left;
                            }

                            th {
                                background: #edf2ff;
                            }

                            img {
                                max-width: 100%;
                                max-height: 400px;
                                object-fit: contain;
                                margin: 12px 0;
                                border: 1px solid #ddd;
                                border-radius: 8px;
                            }

                            section {
                                margin-top: 30px;
                                page-break-inside: avoid;
                            }

                            .notice {
                                background: #fff5df;
                                padding: 15px;
                                border-radius: 8px;
                                line-height: 1.6;
                            }

                            button {
                                padding: 12px 20px;
                                background: #294f9e;
                                color: white;
                                border: 0;
                                border-radius: 6px;
                                cursor: pointer;
                            }

                            @media print {
                                button {
                                    display: none;
                                }
                            }

                        </style>

                    </head>

                    <body>

                        <h1>OncoLens Prediction Report</h1>

                        <p>
                            Generated: ${new Date().toLocaleString()}
                        </p>

                        <h2>Original Uploaded Image</h2>

                        <img src="${originalImage}" alt="Original image">

                        <h2>Model Predictions</h2>

                        <table>

                            <thead>

                                <tr>
                                    <th>Model</th>
                                    <th>Prediction</th>
                                    <th>Confidence</th>
                                    <th>Cancer Probability</th>
                                </tr>

                            </thead>

                            <tbody>
                                ${modelRows}
                            </tbody>

                        </table>

                        <h2>Overall Comparison</h2>

                        <p>
                            Majority prediction:
                            ${escapeHTML(comparison.majority_prediction || "N/A")}
                        </p>

                        <p>
                            Model agreement:
                            ${escapeHTML(comparison.agreement || "N/A")}
                        </p>

                        <p>
                            Average cancer probability:
                            ${comparison.average_cancer_probability ?? "N/A"}%
                        </p>

                        <p>
                            Probability range:
                            ${comparison.probability_range ?? "N/A"} percentage points
                        </p>

                        <p>
                            Cancer votes:
                            ${comparison.cancer_votes ?? "N/A"}
                        </p>

                        <p>
                            Non-Cancer votes:
                            ${comparison.non_cancer_votes ?? "N/A"}
                        </p>

                        <h2>Grad-CAM Visualizations</h2>

                        ${heatmapSections}

                        <div class="notice">

                            <strong>Research Disclaimer</strong>

                            <p>
                                OncoLens is an academic research prototype.
                                Its predictions and Grad-CAM visualizations
                                are experimental and have not been clinically
                                validated. They must not be used for medical
                                diagnosis or treatment decisions.
                            </p>

                            <p>
                                Grad-CAM visualizations show relative
                                model feature attribution. They do not
                                identify or confirm cancerous tissue.
                            </p>

                        </div>

                        <br>

                        <button onclick="window.print()">
                            Print / Save as PDF
                        </button>

                    </body>
                    </html>
                `;


                reportWindow.document.open();

                reportWindow.document.write(
                    reportHTML
                );

                reportWindow.document.close();

                setMessage(
                    "Report generated. Use Print / Save as PDF in the new tab."
                );

            } catch (error) {

                console.error(error);

                setMessage(
                    error.message || "Unable to generate report.",
                    true
                );

            } finally {

                downloadReportBtn.disabled = false;

                downloadReportBtn.textContent =
                    "Download Report";

            }

        }
    );


    // ==================================================
    // HTML ESCAPING
    // ==================================================

    function escapeHTML(value) {

        return String(value ?? "").replace(
            /[&<>"']/g,
            function (character) {

                const replacements = {
                    "&": "&amp;",
                    "<": "&lt;",
                    ">": "&gt;",
                    '"': "&quot;",
                    "'": "&#39;"
                };

                return replacements[character];

            }
        );

    }

});
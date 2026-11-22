
document.addEventListener("DOMContentLoaded", function () {

    const chartCanvas = document.getElementById("predictionChart");

    // Grafik hanya dibuat setelah pengguna menjalankan prediksi.
    if (!chartCanvas) {
        return;
    }

    const labels = [
        ...historyLabels,
        targetLabel
    ];

    const actualHistory = [
        ...historyValues,
        null
    ];

    const predictedHistory = [
        ...Array(historyValues.length).fill(null),
        predictedValue
    ];

    // Nilai aktual target hanya tersedia untuk waktu historis.
    const actualTarget = [
        ...Array(historyValues.length).fill(null),
        actualValue
    ];

    new Chart(chartCanvas, {
        type: "line",

        data: {
            labels: labels,

            datasets: [
                {
                    label: "PM2.5 Historis",
                    data: actualHistory,
                    borderColor: "#2563eb",
                    backgroundColor: "rgba(37, 99, 235, 0.08)",
                    borderWidth: 2,
                    pointRadius: 2,
                    tension: 0.3,
                    fill: true
                },

                {
                    label: "PM2.5 Prediksi",
                    data: predictedHistory,
                    borderColor: "#7c3aed",
                    backgroundColor: "#7c3aed",
                    pointRadius: 7,
                    pointHoverRadius: 9,
                    borderWidth: 3,
                    showLine: false
                },

                {
                    label: "PM2.5 Aktual Target",
                    data: actualTarget,
                    borderColor: "#16a34a",
                    backgroundColor: "#16a34a",
                    pointRadius: 7,
                    pointHoverRadius: 9,
                    borderWidth: 3,
                    showLine: false
                }
            ]
        },

        options: {
            responsive: true,
            maintainAspectRatio: false,

            interaction: {
                intersect: false,
                mode: "index"
            },

            plugins: {
                legend: {
                    position: "bottom",
                    labels: {
                        usePointStyle: true,
                        padding: 20,
                        font: {
                            family: "DM Sans",
                            size: 11
                        }
                    }
                },

                tooltip: {
                    callbacks: {
                        label: function(context) {
                            if (context.raw === null) {
                                return null;
                            }

                            return context.dataset.label +
                                ": " +
                                Number(context.raw).toFixed(2) +
                                " µg/m³";
                        }
                    }
                }
            },

            scales: {
                x: {
                    ticks: {
                        maxTicksLimit: 9,
                        font: {
                            size: 10
                        }
                    },

                    grid: {
                        display: false
                    }
                },

                y: {
                    beginAtZero: true,

                    title: {
                        display: true,
                        text: "PM2.5 (µg/m³)"
                    },

                    ticks: {
                        font: {
                            size: 10
                        }
                    }
                }
            }
        }
    });

});

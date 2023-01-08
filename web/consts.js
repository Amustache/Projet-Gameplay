const FPS           = 60
const DEFAULT_VALUE = "UP"
const VIDEO         = getById("video")
const UPDATE_TIME   = 2000
const GRAPH_DECIMATE_FACTOR = 15

const KEYS = [
    "Key.right",
    "Key.left",
    "x"
]

const CHART_OPTIONS = {
    showLine: true,
    pointRadius: 2,
    pointHoverRadius: 2,
    animation: { duration: 0 },
    plugins: { 
        annotation: { 
            annotations: {
                timeline: {
                    type: 'line',
                    xMax: 0,
                    xMin: 0,
                    borderColor: 'rgb(255, 99, 132)'
                }
            } 
        },
        zoom: { 
            zoom: {
                wheel: { enabled: true },
                pinch: { enabled: true },
                mode: 'x',
            },
            pan: {
                enabled: true,
                mode: 'x',
            }
        }
    }
}
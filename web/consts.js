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

const chart_plugins = { 
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

const CHART_OPTIONS = {
    showLine: true,
    animation: { duration: 0 },
    plugins: chart_plugins
}
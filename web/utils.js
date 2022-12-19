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
    } ,
    zoom: {
        zoom: {
            wheel: {
                enabled: true,
            },
            pinch: {
                enabled: true
            },
            mode: 'x',
        },
        pan: {
            enabled: true,
            mode: 'x',
        },
    }
}
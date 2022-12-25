class KeyLog{
    constructor(input, keys){
        this.raw_input   = input
        this.length      = input.length
        this.keys        = keys
        this.parsed_data = undefined
        this.ttable      = undefined
    }

    getEntry(index){
        if(index >= this.length){
            return undefined
        }else{
            return {
                index: index,
                frame: this.raw_input[index][0],
                key:   this.raw_input[index][1],
                state: this.raw_input[index][2]
            }
        }
    }

    getByFrame(){
        if(this.parsed_data != undefined){
            return this.parsed_data
        }else{
            this.parsed_data = []
            let current_state = {}
            let entry = this.getEntry(0)

            this.keys.forEach(e => current_state[e] = DEFAULT_VALUE)
        
            for(let frame=0; frame<VIDEO.duration*FPS; frame++){
                while(entry != undefined && entry.frame == frame){
                    if(this.keys.includes(entry.key)){
                        current_state[entry.key] = entry.state
                    }
                    entry = this.getEntry(entry.index+1)
                }
                this.parsed_data.push({...current_state}) // Make a copy
            }
            return this.parsed_data
        }
    }

    getCurrentFrame(){
        return this.getByFrame()[getCurrentFrame()]
    }

    getTransitionTables(){
        if(this.ttable != undefined){
            return this.ttable
        }else{
            this.ttable = []
            let state = []
            for(let i=0; i<this.keys.length; i++){
                state.push([])
                for(let j=0; j<this.keys.length; j++){
                    state[i].push(0)
                }
            }

            let prev_index = this.keys.indexOf(this.getEntry(0).key)
            for(let i=0; i<this.length; i++){
                let entry = this.getEntry(i)
                if(this.keys.includes(entry.key) && entry.state == "DOWN"){
                    let index = this.keys.indexOf(entry.key)
                    state[index][prev_index] += 1
                    this.ttable.push({
                        state : JSON.parse(JSON.stringify(state)), // Deep copy
                        frame : entry.frame
                    })
                    prev_index = index
                }
            }
            return this.ttable
        }
    }

    getTransitionTableOfFrame(frame){
        let index = 0
        let ttables = this.getTransitionTables()
        while(index < ttables.length && ttables[index].frame <= frame){
            index++
        }
        return ttables[index]
    }

    get_ttable_row_sums(ttable){
        let sum = Array(this.keys.length).fill(0)
        ttable.state.forEach(row => {
            for(let i=0; i<row.length; i++){
                sum[i] += row[i]
            }
        })
        return sum
    }

    print_ttable(table_html){
        let ttable = this.getTransitionTableOfFrame(getCurrentFrame()) 
    
        table_html.innerHTML = ""
        let new_content = ""
        let row_sums = this.get_ttable_row_sums(ttable)

        new_content += "<th>"
        this.keys.forEach(key => new_content += "<td>"+key+"</td>")
        new_content += "</th>"
    
        for(let i=0; i < this.keys.length; i++){    
            new_content += "<tr><td>" + KEYS[i] + "</td>"
            for(let j=0; j < this.keys.length; j++){
                let value = (100*ttable.state[i][j]/row_sums[j]).toFixed(1)
                new_content += "<td style='background-color:hsl(0, 100%, "+(100-value/2)+"%)'>" + value + "%</td>"
            }
            table_html.innerHTML += new_content + "</tr>"
            new_content = ""
        }
    }

    get_ttable_graph_dataset(){
        let datasets = []
        let ttables = this.getTransitionTables()

        for(let i=0; i<ttables.length; i+=GRAPH_DECIMATE_FACTOR){
            let ttable = ttables[i]
            let row_sums = this.get_ttable_row_sums(ttable)
            let index = 0
            for(let row=0; row<ttable.state.length; row++){
                for(let col=0; col<ttable.state[row].length; col++){
                    if(datasets[index] == undefined){
                        datasets[index] = {
                            label: this.keys[index%this.keys.length] + "→" + this.keys[Math.floor(index/this.keys.length)],
                            data: [],
                            borderWidth: 1
                        }
                    }
                    datasets[index].data.push({
                        x: ttable.frame/FPS,
                        y: ttable.state[row][col]/row_sums[col]
                    })
                    index++
                }
            }
        }
        return datasets
    }
}
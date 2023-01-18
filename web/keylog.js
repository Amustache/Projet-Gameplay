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

    getGraphDataset(){
        let datasets = []
        let ttables = this.getTransitionTables()

        for(let i=0; i<ttables.length; i+=GRAPH_DECIMATE_FACTOR){
            let ttable = ttables[i]
            let row_sums = getTTableRowSums(ttable, this.keys)
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
                        x: ttable.frame,
                        y: 100*ttable.state[row][col]/row_sums[col]
                    })
                    index++
                }
            }
        }
        return datasets
    }
}
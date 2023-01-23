class KeyLog{
    constructor(input, keys, offset=0){
        this.rawInput = input
        this.keys     = keys
        this.offset   = offset

        this.byFrame = []
        {
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
                this.byFrame.push({...current_state}) // Make a copy
            }
        }

        this.ttable = []
        {
            let state = []
            for(let i=0; i<this.keys.length; i++){
                state.push([])
                for(let j=0; j<this.keys.length; j++){
                    state[i].push(0)
                }
            }

            let prev_index = this.keys.indexOf(this.getEntry(0).key)
            for(let i=0; i<this.rawInput.length; i++){
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
        }
    }

    getEntry(index){
        if(index >= this.rawInput.length){
            return undefined
        }
        let index_corrected = Math.min(index+this.offset, this.rawInput.length-1)
        return {
            index: index,
            frame: parseInt(this.rawInput[index_corrected].FRAME),
            key:   this.rawInput[index_corrected].KEY,
            state: this.rawInput[index_corrected].STATUS
        }
    }

    getAllFrames(){
        return this.byFrame
    }

    getFrame(frame){
        return this.byFrame[frame]
    }

    getTransitionTables(){
        return this.ttable
    }

    getTransitionTable(frame){
        let index = 0
        while(index < this.ttable.length && this.ttable[index].frame <= frame){
            index++
        }
        return this.ttable[index]
    }

    getGraphDataset(){
        let dataset = []
        this.ttable.forEach(ttable => {
            let row_sums = getTTableRowSums(ttable, this.keys)
            let index = 0
            ttable.state.forEach(row => {
                for(let col=0; col<row.length; col++){
                    let label = this.keys[index%this.keys.length] + "→" + this.keys[Math.floor(index/this.keys.length)]
                    dataset.push({
                        frame: ttable.frame,
                        y: 100*row[col]/row_sums[col],
                        key: label
                    })
                    index++
                }
            })
        })
        return dataset
    }
}
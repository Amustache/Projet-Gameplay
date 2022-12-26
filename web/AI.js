class AI{
    constructor(ttable, keys){
        this.keys = keys
        this.ttable = []
        this.sums = get_ttable_row_sums({state:ttable}, this.keys)

        for(let i=0; i<ttable.length; i++){
            this.ttable.push([])
            for(let j=0; j<ttable[i].length; j++){
                this.ttable[i].push({
                    count: ttable[i][j],
                    probability: ttable[i][j]/this.sums[j]
                })
            }
        }
    }

    find_loop_in_ttable(){
        let result = []
        for(let i=0; i<this.ttable.length; i++){
            let loops = this.find_loop_in_table_entry(i)
            let best_loop = {heuristic:0, path:[]}
            loops.forEach(loop => {
                if(loop.heuristic > best_loop.heuristic){
                    best_loop = loop
                }
            })
            result.push(best_loop)
        }
        return result.sort((a,b) => a.heuristic < b.heuristic)
    }

    find_loop_in_table_entry(entry_index){
        let graph = {
            entry_index : entry_index,
            visited_indices: [],
            candidates: [{
                path:[entry_index],
                heuristic: this.sums[entry_index],
                visited: false
            }],
            best_index: 0
        }
    
        let loops = []
        while(graph.best_index != -1){
    
            let best_candidate = graph.candidates[graph.best_index]
            graph.visited_indices.push(best_candidate.path.at(-1))
            this.get_next_candidates(best_candidate).forEach(candidate => {
                let last_index = candidate.path.at(-1)
                if(last_index == entry_index){
                    loops.push(candidate)
                }else if(!this._was_visited(graph, last_index)){
                    graph.candidates.push(candidate) 
                }
            })
            best_candidate.visited = true

            graph.best_index = -1
            for(let i=0; i<graph.candidates.length; i++){
                let candidate = graph.candidates[i]
                if(!candidate.visited && !this._was_visited(graph, candidate.path.at(-1))){
                    if(graph.best_index == -1 || candidate.heuristic > graph.candidates[graph.best_index].heuristic ){
                        graph.best_index = i
                    }
                }
            }
        }
        return loops
    }

    _was_visited(graph, index){
        return graph.visited_indices.indexOf(index) != -1
    }

    get_next_candidates(candidate){
        let result = []
        let entry_index = candidate.path.at(-1)
        for(let i=0; i<this.ttable.length; i++){
            if(this.ttable[i][entry_index] != 0){
                result.push({
                    path: candidate.path.concat([i]),
                    heuristic: candidate.heuristic * this.ttable[i][entry_index].probability,
                    visited: false
                })
            }
        }
        return result
    }
}

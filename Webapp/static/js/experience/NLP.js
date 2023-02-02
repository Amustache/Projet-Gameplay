class NLP {
    constructor(raw_data, keys) {
        this.keys = keys
        this.data = []
        let previous = undefined
        raw_data.forEach(elem => {
            let processed = this.elemToStr(elem)
            if (processed != previous && processed != undefined) {
                this.data.push(processed)
                previous = processed
            }
        })
    }

    elemToStr(element) {
        let result = 0
        for (let i = 0; i < this.keys.length; i++) {
            if (element[this.keys[i]] == "DOWN") {
                result += Math.pow(2, i)
            }
        }
        return result.toString()
    }

    elemParseStr(element) {
        let result = []
        for (let i = 0; i < element.length; i++) {
            let state_num = parseInt(element[i])
            let current_key_state = []
            for (let j = this.keys.length - 1; j >= 0; j--) {
                if (state_num >= Math.pow(2, j)) {
                    current_key_state.push(this.keys[j])
                    state_num -= Math.pow(2, j)
                }
            }
            result.push(current_key_state.length == 0 ? "[None]" : "[" + current_key_state + "]")
        }
        return result
    }

    tokenize(e1, e2) {
        return e1 + '' + e2
    }

    BPE_iteration(data_in) {
        let tokens = {}
        let best = {token: undefined, count: 0}

        for (let i = 0; i < data_in.length - 1; i++) {
            let token = this.tokenize(data_in[i], data_in[i + 1])
            if (tokens[token] == undefined) {
                tokens[token] = 0
            }
            tokens[token]++
            if (tokens[token] > best.count) {
                best.token = token
                best.count = tokens[token]
            }
        }
        let data_out = []
        for (let i = 0; i < data_in.length; i++) {
            if (i != data_in.length - 1 && this.tokenize(data_in[i], data_in[i + 1]) == best.token) {
                data_out.push(best.token)
                i++
            } else {
                data_out.push(data_in[i])
            }
        }
        return [data_out, best]
    }

    BPE(nb_iter = 10) {
        let data = this.data
        let keys = []

        for (let i = 0; i < nb_iter; i++) {
            let [data_new, best_token] = this.BPE_iteration(data)
            data = data_new
            keys = []
            data.forEach(elem => {
                if (keys.indexOf(elem) == -1) {
                    keys.push(elem)
                }
            })
        }

        let keys_result = []
        keys.forEach(key => keys_result.push([key, this.elemParseStr(key)]))

        return [this.computeTTable(data, keys), keys_result]
    }

    computeTTable(data, keys) {
        let state = []
        for (let i = 0; i < keys.length; i++) {
            state[i] = []
            keys.forEach(_ => state[i].push(0))
        }

        let prev_index = keys.indexOf(data[0])
        data.forEach(elem => {
            let index = keys.indexOf(elem)
            state[index][prev_index] += 1
            prev_index = index
        })
        return { state: state }
    }
}

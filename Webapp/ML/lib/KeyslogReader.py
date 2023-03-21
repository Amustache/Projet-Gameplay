import pandas
import torch


class KeyslogReader:
    def __init__(self, keys_path, device, frame_step=1, start_frame=0, offset=0):
        self.data = []

        csv_file = pandas.read_csv(keys_path)

        print(f"Loading keys ({len(csv_file)} entries)...")

        current_state = set()
        current_frame = start_frame
        for i in range(len(csv_file)):
            frame, key, state = csv_file.loc[i]

            while current_frame < frame + offset:
                self.data.append(self.keysToArray(current_state))
                current_frame += frame_step

            if state == "UP" and key in current_state:
                current_state.remove(key)
            elif state == "DOWN":
                current_state.add(key)

        self.data = torch.Tensor(self.data).to(device)

        print(f"Keys loaded. Nb of output keys : {len(self.data)}")

    def __getitem__(self, i):
        if i < len(self.data):
            return self.data[i]
        else:
            return self.data[len(self.data) - 1]

    def keysToArray(self, keys):
        output = [0, 0, 0]
        if "Key.left" in keys:
            output[0] = 1
        if "Key.right" in keys:
            output[1] = 1
        if "'x'" in keys:
            output[2] = 1
        return output

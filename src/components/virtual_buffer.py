import random

class VirtualBuffer:
    def __init__(self, n_batches):
        self.d = {}
        self.n_trajs = n_batches
        self.trajs_in_buffer = 0

    def insert_episode_batch(self, traj, task, original_task):
        if task not in self.d:
            self.d[task] = {task: []}
        space_left = self.n_trajs - self.trajs_in_buffer
        if space_left > 0:
            try:
                self.d[task][original_task].append(traj)
                self.trajs_in_buffer += 1
            except:
                print(task, original_task, self.d[task].keys())
                assert 0
        else:
            most_trajs = 0
            task_with_most_trajs = None
            for big_task, big_buffer in self.d.items():
                n_trajs = sum(len(small_buffer) for small_task, small_buffer in big_buffer.items())
                if n_trajs >= most_trajs:
                    most_trajs = n_trajs
                    task_with_most_trajs = big_task
            small_task_with_most_trajs = max(self.d[task_with_most_trajs].items(), key = lambda x:len(x[1]))[0]
            self.d[task_with_most_trajs][small_task_with_most_trajs] = self.d[task_with_most_trajs][small_task_with_most_trajs][1:]

            self.d[task][original_task].append(traj)
                
        total_ntrajs = 0
        for big_task, big_buffer in self.d.items():
            total_ntrajs += sum(len(small_buffer) for small_task, small_buffer in big_buffer.items())
        assert total_ntrajs <= self.n_trajs

    def can_sample(self, bs):
        can_sample_task_cnt = 0
        for big_task, big_buffer in self.d.items():
            n_batch = sum(len(small_buffer) for small_task, small_buffer in big_buffer.items())
            if n_batch >= bs:
                can_sample_task_cnt += 1
        return can_sample_task_cnt >= 2

    def sample(self, bs):
        d = {}
        for big_task, big_buffer in self.d.items():
            all_data = []
            for small_task, small_bufer in big_buffer.items():
                all_data += small_bufer
            n_batch = len(all_data)
            if n_batch >= bs:
                episode_sample = random.sample(all_data, bs)
                d[big_task] = episode_sample
        return d

    def merge_task(self, merge_task, original_task):
        assert merge_task in self.d
        self.d[merge_task][original_task] = []

    def n_big_buffers(self):
        return len(self.d)

    def n_small_buffers(self):
        return sum(len(big_buffer) for big_task, big_buffer in self.d.items())

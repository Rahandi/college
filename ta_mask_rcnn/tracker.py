import math
from copy import deepcopy

class Tracker:
    def __init__(self):
        '''
        self.space = [
            {
                'id': 0, 
                'coord': [x1, y1, x2, y2], 
                'status': 0 (0 unoccupied, 1 occupied)
            }
        ]

        self.temporary = [
            {
                'age' = 1, 
                'detected' = 1,
                'coord' = [x1, y1, x2, y2], 
            }
        ]
        '''

        self.space = []
        self.temporary = []

        self.same_space_threshold = 25
        self.max_distance = 50
        self.square_size = 50

        self.fps = 20
        self.frame_detect = 10 * self.fps
        self.max_age = 300 * self.fps
        self.max_occupied_age = 10 * self.fps

    def update(self, current):
        '''
        current = [
            [x1, y1, x2, y2]
        ]
        '''
        self._interpolate_data()
        self._check_temporary_space_overlap()
        new_current = self._check_space(current)
        self._check_temporary(new_current)
        self._check_max_age_occupied()

        return (self.space, self.temporary)

    def _get_center(self, coordinate):
        '''
        coordinate = [x1, y1, x2, y2]
        '''
        x = int((coordinate[0] + coordinate[2])/2)
        y = int((coordinate[1] + coordinate[3])/2)
        return x, y

    def _calc_distance(self, x1, y1, x2, y2):
        distance = math.sqrt(math.pow(x1-x2, 2) + math.pow(y1-y2, 2))
        return distance

    def _check_intersection(self, first, second):
        '''
        first = [x1, y1, x2, y2]
        second = [x1, y1, x2, y2]
        '''
        if second[0] < first[0] < second[2]:
            if second[1] < first[1] < second[3] or second[1] < first[3] < second[3]:
                middle = int((first[0] + second[2]) / 2)
                first[0] = middle
                second[2] = middle

        if second[0] < first[2] < second[2]:
            if second[1] < first[1] < second[3] or second[1] < first[3] < second[3]:
                middle = int((first[2] + second[0]) / 2)
                first[2] = middle
                second[0] = middle

        return first, second

    def _check_max_age_occupied(self):
        for i in range(len(self.space)):
            if self.space[i]['status'] == 0:
                if self.space[i]['age'] > 0:
                    self.space[i]['status'] = 1
                    self.space[i]['age'] -= 1
            else:
                self.space[i]['age'] = self.max_occupied_age

    def _check_temporary_space_overlap(self):
        '''
        self.temporary = [
            {
                'age' = 1, 
                'detected' = 1,
                'coord' = [x, y], 
            }
        ]

        self.space = [
            {
                'id': 0, 
                'coord': [x, y], 
                'square': [x1, y1, x2, y2]
                'status': 0 (0 unoccupied, 1 occupied)
            }
        ]
        '''
        for i in range(len(self.temporary)-1, -1, -1):
            self.temporary[i]['age'] += 1
            if self.temporary[i]['age'] >= self.max_age:
                self.temporary.remove(self.temporary[i])
        for i in range(len(self.space)):
            x1, y1, x2, y2 = self.space[i]['square']
            marker = 0
            for j in range(len(self.temporary)-1, -1, -1):
                x, y = self.temporary[j]['coord']

                marker = 1 if x1 < x < x2 and y1 < y < y2 else 0

                if marker == 1:
                    self.space[i]['status'] = 1
                    self.temporary.remove(self.temporary[j])

    def _check_space(self, current):
        '''
        current = [
            [x1, y1, x2, y2]
        ]

        self.space = [
            {
                'id': 0, 
                'coord': [x, y], 
                'square': [x1, y1, x2, y2]
                'status': 0 (0 unoccupied, 1 occupied)
            }
        ]
        '''
        current_tempo = deepcopy(current)
        for i in range(len(self.space)):
            x1, y1, x2, y2 = self.space[i]['square']

            marker = 0 # 0 empty, 1 occupied
            for j in range(len(current_tempo)-1, -1, -1):
                x_current, y_current = self._get_center(current_tempo[j])

                marker = 1 if x1 < x_current < x2 and y1 < y_current < y2 else 0

                if marker == 1:
                    self.space[i]['status'] = 1
                    current_tempo.remove(current_tempo[j])

                    break
            
            if marker == 0:
                self.space[i]['status'] = 0
        
        return deepcopy(current_tempo)

    def _check_temporary(self, current):
        '''
        current = [
            [x1, y1, x2, y2]
        ]

        self.temporary = [
            {
                'age' = 1,
                'detected' = 1,
                'coord' = [x, y], 
            }
        ]
        '''
        
        for i in range(len(current)):
            marker = 0 # (0 new, 1 old)
            
            x_current, y_current = self._get_center(current[i])
            for j in range(len(self.temporary)-1, -1, -1):
                x_temporary, y_temporary = self.temporary[j]['coord']

                distance = self._calc_distance(x_current, y_current, x_temporary, y_temporary)
                marker = 1 if distance <= self.same_space_threshold else 0

                if marker == 1:
                    # update temporary
                    self.temporary[j]['detected'] += 1
                    self.temporary[j]['coord'][0] = int((x_current + x_temporary) / 2)
                    self.temporary[j]['coord'][1] = int((y_current + y_temporary) / 2)

                    if self.temporary[j]['detected'] >= self.frame_detect:
                        temp = {
                            'id': len(self.space), 
                            'coord': self.temporary[j]['coord'], 
                            'square': [self.temporary[j]['coord'][0] - self.square_size, self.temporary[j]['coord'][1] - self.square_size, self.temporary[j]['coord'][0] + self.square_size, self.temporary[j]['coord'][1] + self.square_size], 
                            # 'square': current[i], 
                            'status': 1,
                            'age': self.max_occupied_age
                        }
                        for k in range(len(self.space)):
                            this = self._check_intersection(self.space[k]['square'], temp['square'])
                            self.space[k]['square'] = this[0]
                            temp['square'] = this[1]
                        marker = 0
                        for l in range(len(self.space)):
                            if self.space[l]['square'][0] < self.temporary[j]['coord'][0] < self.space[l]['square'][2] and self.space[l]['square'][1] < self.temporary[j]['coord'][1] < self.space[l]['square'][3]:
                                print(str(self.temporary[j]['coord']) + ' inside' + str(self.space[l]['square']))
                                marker = 1
                        if marker == 0:
                            self.space.append(temp)
                        self.temporary.remove(self.temporary[j])
                    break
            
            if marker == 0:
                temp = {
                    'age': 1, 
                    'detected': 1,
                    'coord': [x_current, y_current]
                }
                self.temporary.append(temp)
                

    def _interpolate_data(self):
        clusters = []

        for item in self.space:
            temp = deepcopy(item)

            if len(clusters) == 0:
                clusters.append([temp])
                continue

            marker = 0
            for cluster in clusters:
                rata_y = sum([x['coord'][1] for x in cluster]) / len(cluster)
                dist = abs(rata_y - item['coord'][1])

                if dist < self.max_distance:
                    cluster.append(temp)
                    marker = 1
                    break

            if marker == 1:
                continue
            else:
                clusters.append([temp])

        for item in clusters:
            item = sorted(item, key=lambda x: x['coord'][0])

            for i in range(1, len(item)):
                if((item[i]['coord'][0] - item[i-1]['coord'][0]) >= (4*self.square_size)):
                    empty_space = (item[i]['coord'][0] - self.square_size) - (item[i-1]['coord'][0] + self.square_size)
                    # print(item[i-1]['id'], item[i]['id'], empty_space)
                    space_avail = math.floor(empty_space / (self.square_size*2))
                    lot_space = math.floor(empty_space / space_avail)
                    for j in range(1, space_avail+1):
                        middle =  [item[i-1]['coord'][0] + (lot_space * j), item[i-1]['coord'][1]]
                        temp = {
                            'id': len(self.space), 
                            'coord': middle, 
                            'square': [middle[0] - self.square_size, middle[1] - self.square_size, middle[0] + self.square_size, middle[1] + self.square_size], 
                            'status': 0,
                            'age': 0
                        }
                        for l in range(len(self.space)):
                            if self.space[l]['square'][0] < middle[0] < self.space[l]['square'][2] and self.space[l]['square'][1] < middle[1] < self.space[l]['square'][3]:
                                print(str(middle) + ' inside' + str(self.space[l]['square']))
                            else:
                                self.space.append(temp)
                                break
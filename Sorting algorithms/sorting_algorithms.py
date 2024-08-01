import time
import pygame
import random
from copy import deepcopy

class Algorithm:
    def __init__(self, name):
        self.array = random.sample(range(512), 512)
        self.original_array = deepcopy(self.array)
        self.name = name
        self.display = None
        self.dimensions = (800, 600)

    def display_setup(self):
        pygame.init()
        self.display = pygame.display.set_mode(self.dimensions)
        pygame.display.set_caption("Sorting Algorithm Visualizer")

    def update_display(self, swap1=None, swap2=None):
        if self.display is None:
            raise RuntimeError("Display not initialized. Call display_setup() first.")

        self.display.fill((48, 48, 48))
        k = self.dimensions[0] // len(self.original_array)
        for i, value in enumerate(self.original_array):
            color = (93, 173, 226)
            if swap1 == i:
                color = (255, 0, 0)
            elif swap2 == i:
                color = (0, 255, 0)
            pygame.draw.rect(self.display, color, (i * k, self.dimensions[1] - value, k, value))
        pygame.display.update()

    def keep_open(self, time_elapsed):
        pygame.display.set_caption(f"Sorting Algorithm: {self.name} | Time: {time_elapsed:.6f} seconds | Status: Done!")
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            pygame.display.update()
        pygame.quit()

    def run(self):
        self.display_setup()
        sorting_time = self.time_sorting()
        self.visualize_sorting()
        self.keep_open(sorting_time)
        return sorting_time

    def time_sorting(self):
        self.sorted_array = deepcopy(self.array)
        start_time = time.perf_counter()
        self.algorithm()
        sorting_time = time.perf_counter() - start_time
        return sorting_time

    def visualize_sorting(self):
        self.original_array = deepcopy(self.array)
        pass

class SelectionSort(Algorithm):
    def __init__(self):
        super().__init__("SelectionSort")

    def algorithm(self):
        n = len(self.sorted_array)
        for i in range(n):
            min_idx = i
            for j in range(i + 1, n):
                if self.sorted_array[j] < self.sorted_array[min_idx]:
                    min_idx = j
            if min_idx != i:
                self.sorted_array[i], self.sorted_array[min_idx] = self.sorted_array[min_idx], self.sorted_array[i]

    def visualize_sorting(self):
        self.original_array = deepcopy(self.array)
        n = len(self.original_array)
        for i in range(n):
            min_idx = i
            for j in range(i + 1, n):
                if self.original_array[j] < self.original_array[min_idx]:
                    min_idx = j
            if min_idx != i:
                self.original_array[i], self.original_array[min_idx] = self.original_array[min_idx], self.original_array[i]
                self.update_display(i, min_idx)
                pygame.time.delay(10)


class BubbleSort(Algorithm):
    def __init__(self):
        super().__init__("BubbleSort")

    def algorithm(self):
        n = len(self.sorted_array)
        for i in range(n):
            for j in range(0, n - i - 1):
                if self.sorted_array[j] > self.sorted_array[j + 1]:
                    self.sorted_array[j], self.sorted_array[j + 1] = self.sorted_array[j + 1], self.sorted_array[j]

    def visualize_sorting(self):
        self.original_array = deepcopy(self.array)
        n = len(self.original_array)
        for i in range(n):
            for j in range(0, n - i - 1):
                if self.original_array[j] > self.original_array[j + 1]:
                    self.original_array[j], self.original_array[j + 1] = self.original_array[j + 1], self.original_array[j]
                    self.update_display(j, j + 1)
                    pygame.time.delay(0)

class InsertionSort(Algorithm):
    def __init__(self):
        super().__init__("InsertionSort")

    def algorithm(self):
        n = len(self.sorted_array)
        for i in range(1, n):
            key = self.sorted_array[i]
            j = i - 1
            while j >= 0 and key < self.sorted_array[j]:
                self.sorted_array[j + 1] = self.sorted_array[j]
                j -= 1
            self.sorted_array[j + 1] = key

    def visualize_sorting(self):
        self.original_array = deepcopy(self.array)
        n = len(self.original_array)
        for i in range(1, n):
            key = self.original_array[i]
            j = i - 1
            while j >= 0 and key < self.original_array[j]:
                self.original_array[j + 1] = self.original_array[j]
                j -= 1
            self.original_array[j + 1] = key
            self.update_display(j + 1, i)
            pygame.time.delay(10)

class MergeSort(Algorithm):
    def __init__(self):
        super().__init__("MergeSort")

    def algorithm(self):
        self.sorted_array = self.merge_sort(self.sorted_array)

    def merge_sort(self, arr):
        if len(arr) > 1:
            mid = len(arr) // 2
            L = arr[:mid]
            R = arr[mid:]
            self.merge_sort(L)
            self.merge_sort(R)
            i = j = k = 0
            while i < len(L) and j < len(R):
                if L[i] < R[j]:
                    arr[k] = L[i]
                    i += 1
                else:
                    arr[k] = R[j]
                    j += 1
                k += 1
            while i < len(L):
                arr[k] = L[i]
                i += 1
                k += 1
            while j < len(R):
                arr[k] = R[j]
                j += 1
                k += 1
        return arr

    def visualize_sorting(self):
        self.original_array = deepcopy(self.array)
        self.original_array = self.merge_sort(self.original_array)
        self.update_display()
        pygame.time.delay(10)

class QuickSort(Algorithm):
    def __init__(self):
        super().__init__("QuickSort")

    def algorithm(self):
        self.sorted_array = self.quick_sort(self.sorted_array)

    def quick_sort(self, arr):
        if len(arr) <= 1:
            return arr
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        return self.quick_sort(left) + middle + self.quick_sort(right)

    def visualize_sorting(self):
        self.original_array = deepcopy(self.array)
        self.original_array = self.quick_sort(self.original_array)
        self.update_display()
        pygame.time.delay(10)

class HeapSort(Algorithm):
    def __init__(self):
        super().__init__("HeapSort")

    def algorithm(self):
        n = len(self.sorted_array)
        for i in range(n // 2 - 1, -1, -1):
            self.heapify(self.sorted_array, n, i)
        for i in range(n - 1, 0, -1):
            self.sorted_array[i], self.sorted_array[0] = self.sorted_array[0], self.sorted_array[i]
            self.heapify(self.sorted_array, i, 0)

    def heapify(self, arr, n, i):
        largest = i
        l = 2 * i + 1
        r = 2 * i + 2
        if l < n and arr[i] < arr[l]:
            largest = l
        if r < n and arr[largest] < arr[r]:
            largest = r
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            self.heapify(arr, n, largest)

    def visualize_sorting(self):
        self.original_array = deepcopy(self.array)
        n = len(self.original_array)
        for i in range(n // 2 - 1, -1, -1):
            self.heapify(self.original_array, n, i)
            self.update_display()
            pygame.time.delay(10)
        for i in range(n - 1, 0, -1):
            self.original_array[i], self.original_array[0] = self.original_array[0], self.original_array[i]
            self.heapify(self.original_array, i, 0)
            self.update_display(i, 0)
            pygame.time.delay(10)

class TimSort(Algorithm):
    def __init__(self):
        super().__init__("TimSort")

    def algorithm(self):
        self.tim_sort(self.sorted_array)

    def insertion_sort(self, array, left, right):
        for i in range(left + 1, right + 1):
            key_item = array[i]
            j = i - 1
            while j >= left and array[j] > key_item:
                array[j + 1] = array[j]
                j -= 1
            array[j + 1] = key_item

    def merge(self, array, l, m, r):
        len1, len2 = m - l + 1, r - m
        left, right = [], []
        for i in range(0, len1):
            left.append(array[l + i])
        for i in range(0, len2):
            right.append(array[m + 1 + i])

        i, j, k = 0, 0, l
        while i < len1 and j < len2:
            if left[i] <= right[j]:
                array[k] = left[i]
                i += 1
            else:
                array[k] = right[j]
                j += 1
            k += 1

        while i < len1:
            array[k] = left[i]
            i += 1
            k += 1

        while j < len2:
            array[k] = right[j]
            j += 1
            k += 1

    def tim_sort(self, array):
        min_run = 32
        n = len(array)
        for i in range(0, n, min_run):
            self.insertion_sort(array, i, min((i + min_run - 1), (n - 1)))

        size = min_run
        while size < n:
            for left in range(0, n, 2 * size):
                mid = min((left + size - 1), (n - 1))
                right = min((left + 2 * size - 1), (n - 1))
                if mid < right:
                    self.merge(array, left, mid, right)
            size = 2 * size

    def visualize_sorting(self):
        self.original_array = deepcopy(self.array)
        min_run = 32
        n = len(self.original_array)

        # Insertion sort for small subarrays
        for i in range(0, n, min_run):
            for j in range(i + 1, min(i + min_run, n)):
                key_item = self.original_array[j]
                k = j - 1
                while k >= i and self.original_array[k] > key_item:
                    self.original_array[k + 1] = self.original_array[k]
                    k -= 1
                self.original_array[k + 1] = key_item
                self.update_display()
                pygame.time.delay(10)

        # Merge sorted subarrays
        size = min_run
        while size < n:
            for left in range(0, n, 2 * size):
                mid = min((left + size - 1), (n - 1))
                right = min((left + 2 * size - 1), (n - 1))
                if mid < right:
                    len1, len2 = mid - left + 1, right - mid
                    left_part, right_part = [], []
                    for i in range(0, len1):
                        left_part.append(self.original_array[left + i])
                    for i in range(0, len2):
                        right_part.append(self.original_array[mid + 1 + i])

                    i, j, k = 0, 0, left
                    while i < len1 and j < len2:
                        if left_part[i] <= right_part[j]:
                            self.original_array[k] = left_part[i]
                            i += 1
                        else:
                            self.original_array[k] = right_part[j]
                            j += 1
                        k += 1

                    while i < len1:
                        self.original_array[k] = left_part[i]
                        i += 1
                        k += 1

                    while j < len2:
                        self.original_array[k] = right_part[j]
                        j += 1
                        k += 1
                    self.update_display()
                    pygame.time.delay(100)
            size = 2 * size




class ShellSort(Algorithm):
    def __init__(self):
        super().__init__("ShellSort")

    def algorithm(self):
        n = len(self.sorted_array)
        gap = n // 2
        while gap > 0:
            for i in range(gap, n):
                temp = self.sorted_array[i]
                j = i
                while j >= gap and self.sorted_array[j - gap] > temp:
                    self.sorted_array[j] = self.sorted_array[j - gap]
                    j -= gap
                self.sorted_array[j] = temp
            gap //= 2

    def visualize_sorting(self):
        self.original_array = deepcopy(self.array)
        n = len(self.original_array)
        gap = n // 2
        while gap > 0:
            for i in range(gap, n):
                temp = self.original_array[i]
                j = i
                while j >= gap and self.original_array[j - gap] > temp:
                    self.original_array[j] = self.original_array[j - gap]
                    j -= gap
                self.original_array[j] = temp
                self.update_display(j, i)
                pygame.time.delay(2)
            gap //= 2

class CombSort(Algorithm):
    def __init__(self):
        super().__init__("CombSort")

    def algorithm(self):
        n = len(self.sorted_array)
        gap = n
        shrink = 1.3
        sorted_ = False
        while not sorted_:
            gap = int(gap / shrink)
            if gap <= 1:
                gap = 1
                sorted_ = True
            i = 0
            while i + gap < n:
                if self.sorted_array[i] > self.sorted_array[i + gap]:
                    self.sorted_array[i], self.sorted_array[i + gap] = self.sorted_array[i + gap], self.sorted_array[i]
                    sorted_ = False
                i += 1

    def visualize_sorting(self):
        self.original_array = deepcopy(self.array)
        n = len(self.original_array)
        gap = n
        shrink = 1.3
        sorted_ = False
        while not sorted_:
            gap = int(gap / shrink)
            if gap <= 1:
                gap = 1
                sorted_ = True
            i = 0
            while i + gap < n:
                if self.original_array[i] > self.original_array[i + gap]:
                    self.original_array[i], self.original_array[i + gap] = self.original_array[i + gap], self.original_array[i]
                    sorted_ = False
                i += 1
                self.update_display(i, i + gap)
                pygame.time.delay(2)

class RadixSort(Algorithm):
    def __init__(self):
        super().__init__("RadixSort")

    def algorithm(self):
        self.sorted_array = self.radix_sort(self.sorted_array)

    def counting_sort(self, arr, exp):
        n = len(arr)
        output = [0] * n
        count = [0] * 10
        for i in range(n):
            index = arr[i] // exp
            count[index % 10] += 1
        for i in range(1, 10):
            count[i] += count[i - 1]
        i = n - 1
        while i >= 0:
            index = arr[i] // exp
            output[count[index % 10] - 1] = arr[i]
            count[index % 10] -= 1
            i -= 1
        for i in range(n):
            arr[i] = output[i]

    def radix_sort(self, arr):
        max_ = max(arr)
        exp = 1
        while max_ // exp > 0:
            self.counting_sort(arr, exp)
            exp *= 10
        return arr

    def visualize_sorting(self):
        self.original_array = deepcopy(self.array)
        max_ = max(self.original_array)
        exp = 1
        while max_ // exp > 0:
            n = len(self.original_array)
            output = [0] * n
            count = [0] * 10
            for i in range(n):
                index = self.original_array[i] // exp
                count[index % 10] += 1
            for i in range(1, 10):
                count[i] += count[i - 1]
            i = n - 1
            while i >= 0:
                index = self.original_array[i] // exp
                output[count[index % 10] - 1] = self.original_array[i]
                count[index % 10] -= 1
                i -= 1
            for i in range(n):
                self.original_array[i] = output[i]
                self.update_display()
                pygame.time.delay(10)
            exp *= 10



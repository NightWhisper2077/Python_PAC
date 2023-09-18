class Queue(list):
    """The Queue class represents a queue.
    enqueue adds the value element to the end of the queue.
    dequeue deletes and returns the first element of the queue.
    the size and clear functions are needed to return the size and clear the queue, respectively"""
    queue = []

    def __init__(self):
        super().__init__()

    def enqueue(self, a):
        self.queue.append(a)

    def dequeue(self):
        return self.queue.pop(0)

    def size(self):
        return len(self.queue)

    def clear(self):
        self.queue.clear()

class Stack(list):
    """The Stack class represents stack.
    push adds the value element to the end of the queue.
    pop deletes and returns the last element of the stack.
    the size and clear functions are needed to return the size and clear the stack, respectively"""
    stack = []

    def __init__(self):
        super().__init__()

    def push(self, a):
        self.stack.append(a)

    def pop(self):
        self.stack.pop()

    def clear(self):
        self.stack.clear()

    def size(self):
        return len(self.stack)

class Broker():
    """The Broker class is needed to work with the queue and call the enqueue and dequeue functions.
    The function get_list returns a queue representation as an independent list"""
    def __init__(self):
        super().__init__()
        self._queue = Queue()

    def enqueue(self, value):
        self._queue.enqueue(value)

    def dequeue(self):
        return self._queue.dequeue()

    def get_list(self):
        list_me = []
        for i in self._queue.queue:
            list_me.append(i)

        return list_me
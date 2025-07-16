class person:
    def __call__(self, name):
        print("__call__"+" Hello " + name)
    def hello(self, name):
        print("Hello " + name)

person = person()
person.hello("<NAME>")
person("zhanglejing")
person()
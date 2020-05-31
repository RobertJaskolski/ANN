import os
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserIconView
from run import forOneAudio, nn


class MyWidget(BoxLayout):
    def __init__(self, **kwargs):
        super(MyWidget, self).__init__(**kwargs)

        container = BoxLayout(orientation='vertical')

        filechooser = FileChooserIconView()
        filechooser.bind(on_selection=lambda x: self.selected(filechooser.selection))

        open_btn = Button(text='open', size_hint=(1, .2))
        open_btn.bind(on_release=lambda x: self.open(filechooser.path, filechooser.selection))

        container.add_widget(filechooser)
        container.add_widget(open_btn)
        self.add_widget(container)

    def open(self, path, filename):
        file = os.path.join(path, filename[0])
        if file.endswith('.ogg'):
            print(forOneAudio(network, file, data))
        else:
            print('Invalid file')


class MyApp(App):
    def build(self):
        return MyWidget()


if __name__ == '__main__':
    network, data = nn()
    MyApp().run()

import os
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.popup import Popup
from run import forOneAudio, nn


def open_file(path, filename):
    file = os.path.join(path, filename[0])

    if file.endswith('.ogg'):
        show_popup(title="Result", text=forOneAudio(network, file, data))
    else:
        show_popup(title="Error", text="Invalid file")


def show_popup(title, text):
    layout = GridLayout(cols=1, padding=10)
    popup_label = Label(text=text)
    close_button = Button(text="OK")

    layout.add_widget(popup_label)
    layout.add_widget(close_button)

    popup_window = Popup(title=title, content=layout, size_hint=(None, None), size=(400, 400),
                         pos_hint={"center_x": 0.5, "center_y": .5})
    popup_window.open()
    close_button.bind(on_press=popup_window.dismiss)


class FileSelect(BoxLayout):
    def __init__(self, **kwargs):
        super(FileSelect, self).__init__(**kwargs)

        container = BoxLayout(orientation='vertical')

        file_chooser = FileChooserIconView()
        file_chooser.bind(on_selection=lambda x: self.selected(file_chooser.selection))

        open_btn = Button(text='Open selected file', size_hint=(1, .2))
        open_btn.bind(on_release=lambda x: open_file(file_chooser.path, file_chooser.selection))

        container.add_widget(file_chooser)
        container.add_widget(open_btn)
        self.add_widget(container)


class ANN(App):
    def build(self):
        return FileSelect()


if __name__ == '__main__':
    network, data = nn()
    ANN().run()

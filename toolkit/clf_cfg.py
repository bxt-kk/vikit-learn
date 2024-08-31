from typing import Dict, Any, Callable
from dataclasses import dataclass
import json

import flet as ft


@dataclass
class ParamWdiget:

    widget:        ft.Control
    transform:     Callable | None=None
    disable_value: Any=None

    def get_value(self) -> Any:
        if self.widget.disabled:
            return self.disable_value
        value = self.widget.value
        if self.transform is not None:
            value = self.transform(value)
        return value


class ItemsBase(ft.ListTile):

    def __init__(self):
        super().__init__()

        self._param_dict:Dict[str, ParamWdiget] = dict()

    def add_param(
            self,
            name:          str,
            transform:     Callable,
            widget:        ft.Control,
            disable_value: Any=None,
        ):

        self._param_dict[name] = ParamWdiget(
            widget=widget,
            transform=transform,
            disable_value=disable_value,
        )

    def get_param_widget(self, name:str) -> ft.Control:
        return self._param_dict[name].widget

    def get_param_value(self, name:str) -> Any:
        return self._param_dict[name].get_value()

    def get_params(self) -> Dict[str, Any]:
        return {
            name: self.get_param_value(name)
            for name in self._param_dict.keys()}

    @classmethod
    def add_to_page(
            clf,
            page:        ft.Page,
            title:       str,
            init_expand: bool,
            padding:     int,
        ) -> 'ItemsBase':

        items = clf()
        page.add(ft.ExpansionTile(
            title=ft.Text(title),
            initially_expanded=init_expand,
            controls_padding=padding,
            controls=[items]))
        return items

    @classmethod
    def add_to_column(
            clf,
            column:      ft.Column,
            title:       str,
            init_expand: bool,
            padding:     int,
        ) -> 'ItemsBase':

        items = clf()
        column.controls.append(ft.ExpansionTile(
            title=ft.Text(title),
            initially_expanded=init_expand,
            controls_padding=padding,
            controls=[items]))
        return items


class TaskItems(ItemsBase):

    def __init__(self):
        super().__init__()

        self.add_param('device', str, widget=ft.Dropdown(
            label='device',
            value='auto',
            options=[ft.dropdown.Option(opt) for opt in (
                'auto',
                'cpu',
                'cuda',
            )]))

        self.add_param('metric_start_epoch', int, widget=ft.TextField(
            label='metric start epoch',
            value='0',
            input_filter=ft.NumbersOnlyInputFilter()))

        self.add_param('fit_features_start', int, disable_value=-1, widget=ft.TextField(
            label='start epoch',
            value='0',
            disabled=True,
            input_filter=ft.NumbersOnlyInputFilter()))

        self.title = ft.Column([
            self.get_param_widget('device'),
            self.get_param_widget('metric_start_epoch'),
            ft.Row([
                self.get_param_widget('fit_features_start'),
                ft.Switch(
                    label='fit features',
                    value=False,
                    on_change=self.switch_fit_features),
            ]),
        ])

    def switch_fit_features(self, e:ft.ControlEvent):
        widget = self.get_param_widget('fit_features_start')
        widget.disabled = not e.control.value
        widget.update()


class ModelItems(ItemsBase):

    def __init__(self):
        super().__init__()

        self.add_param('num_scans', int, widget=ft.Slider(
            label='{value}', min=1, max=7, divisions=6, value=3))

        self.add_param('scan_range', int, widget=ft.Slider(
            label='{value}', min=1, max=7, divisions=6, value=4))

        self.add_param('backbone', str, widget=ft.Dropdown(
            label='backbone',
            value='mobilenet_v3_small',
            options=[ft.dropdown.Option(opt) for opt in (
                'mobilenet_v3_small',
                'mobilenet_v3_large',
                'mobilenet_v3_larges',
                'mobilenet_v2',
                'dinov2_vits14_h192',
                'dinov2_vits14',
            )]))

        self.add_param('backbone_pretrained', bool, widget=ft.Switch(
            label='backbone pretrained', value=True))

        self.title = ft.Column([
            ft.Text('num scans'), self.get_param_widget('num_scans'),
            ft.Text('scan range'), self.get_param_widget('scan_range'),
            self.get_param_widget('backbone'),
            self.get_param_widget('backbone_pretrained'),
        ])


class TrainerItems(ItemsBase):

    def __init__(self):
        super().__init__()

        self.add_param('output', str, widget=ft.TextField(
            label='output', value=''))

        self.add_param('checkpoint', lambda s: s or None, widget=ft.TextField(
            label='checkpoint', value=''))

        self.add_param('drop_optim', bool, widget=ft.Switch(
            label='drop checkpoint optim', value=False))

        self.add_param('drop_lr_scheduler', bool, widget=ft.Switch(
            label='drop checkpoint lr scheduler', value=False))

        self.add_param('optim_method', str, widget=ft.Dropdown(
            label='optim method',
            value='AdamW',
            options=[ft.dropdown.Option(opt) for opt in (
                'AdamW',
                'Adam',
                'SGD',
                'NAdam',
            )]))

        self.add_param('lr', float, widget=ft.TextField(
            label='lr', value='1e-3'))

        self.add_param('weight_decay', float, widget=ft.TextField(
            label='weight decay', value='0.01', disabled=True))

        self.add_param('lrf', float, widget=ft.TextField(
            label='lrf', value='0'))

        self.add_param('T_num', float, widget=ft.TextField(
            label='number of T', value='1.0'))

        self.add_param('grad_steps', int, widget=ft.TextField(
            label='grad steps', value='1', input_filter=ft.NumbersOnlyInputFilter()))

        self.add_param('epochs', int, widget=ft.TextField(
            label='epochs', value='1', input_filter=ft.NumbersOnlyInputFilter()))

        self.add_param('show_step', int, widget=ft.TextField(
            label='show step', value='50', input_filter=ft.NumbersOnlyInputFilter()))

        self.add_param('save_epoch', int, widget=ft.TextField(
            label='save_epoch', value='1', input_filter=ft.NumbersOnlyInputFilter()))

        self.title = ft.Column([
            self.get_param_widget('output'),
            self.get_param_widget('checkpoint'),
            self.get_param_widget('drop_optim'),
            self.get_param_widget('drop_lr_scheduler'),
            self.get_param_widget('optim_method'),
            self.get_param_widget('lr'),
            ft.Row([
                self.get_param_widget('weight_decay'),
                ft.Switch(label='use default', value=True, on_change=self.switch_weight_decay)]),
            self.get_param_widget('lrf'),
            self.get_param_widget('T_num'),
            self.get_param_widget('grad_steps'),
            self.get_param_widget('epochs'),
            self.get_param_widget('show_step'),
            self.get_param_widget('save_epoch'),
        ])

    def switch_weight_decay(self, e:ft.ControlEvent):
        widget = self.get_param_widget('weight_decay')
        widget.disabled = e.control.value
        widget.update()


class DatasetItems(ItemsBase):

    def __init__(self):
        super().__init__()

        self.add_param('type', str, widget=ft.Dropdown(
            label='type', value='ImagesFolder', options=[
            ft.dropdown.Option(opt) for opt in (
                'ImagesFolder',
                'OxfordIIITPet',
                'Places365',
            )], on_change=self.dropdown_type))

        self.add_param('extensions',
            transform=lambda s: tuple(s.strip() for s in s.split(',')),
            widget=ft.TextField(label='extensions', value='jpg,jpeg,png'))

        self.add_param('root', str, widget=ft.TextField(
            label='root', value=''))

        self.add_param('transform', str, widget=ft.Dropdown(
            label='transform', value='imagenetx224', options=[
            ft.dropdown.Option(opt) for opt in (
                'imagenetx224',
                'imagenetx256',
                'imagenetx384',
                'imagenetx448',
                'imagenetx512',
                'imagenetx640',
            )]))

        self.add_param('num_workers', int, widget=ft.TextField(
            label='number of workers', value='4',
            input_filter=ft.NumbersOnlyInputFilter()))

        self.add_param('batch_size', int, widget=ft.TextField(
            label='batch size', value='16',
            input_filter=ft.NumbersOnlyInputFilter()))

        self.title = ft.Column([
            self.get_param_widget('type'),
            self.get_param_widget('extensions'),
            self.get_param_widget('root'),
            self.get_param_widget('transform'),
            self.get_param_widget('num_workers'),
            self.get_param_widget('batch_size'),
        ], alignment=ft.alignment.top_left)

    def dropdown_type(self, e:ft.ControlEvent):
        widget = self.get_param_widget('extensions')
        widget.disabled = e.control.value != 'ImagesFolder'
        widget.update()


def main(page:ft.Page):
    page.scroll = ft.ScrollMode.AUTO

    taskItems = TaskItems.add_to_page(page, title='Task', init_expand=False, padding=10)
    modelItems = ModelItems.add_to_page(page, title='Model', init_expand=False, padding=10)
    trainerItems = TrainerItems.add_to_page(page, title='Trainer', init_expand=False, padding=10)
    datasetItems = DatasetItems.add_to_page(page, title='Dataset', init_expand=False, padding=10)

    def dump_configure():
        path = configure_filename.value
        if path == '': return

        cfg = dict(
            task=taskItems.get_params(),
            model=modelItems.get_params(),
            dataset=datasetItems.get_params(),
            trainer=trainerItems.get_params(),
        )
        cfg_json = json.dumps(cfg, ensure_ascii=False, indent=2)
        print(cfg_json)
        with open(path, 'w') as f:
            f.write(cfg_json)

    def pick_files_result(e:ft.FilePickerResultEvent):
        configure_filename.value = e.path
        configure_filename.update()
        dump_configure()

    def save_configure(e:ft.ControlEvent):
        if not configure_filename.value:
            pick_files_dialog.save_file(allowed_extensions=['json'])
            return
        dump_configure()

    pick_files_dialog = ft.FilePicker(on_result=pick_files_result)
    page.overlay.append(pick_files_dialog)

    configure_filename = ft.Text(value='')
    page.add(ft.Row([
        ft.ElevatedButton('Save', icon=ft.icons.SAVE, on_click=save_configure),
        ft.ElevatedButton('Save as', icon=ft.icons.SAVE_AS, on_click=lambda _: pick_files_dialog.save_file(
            allowed_extensions=['json'])),
        configure_filename,
    ]))


def entry():
    ft.app(main)


if __name__ == '__main__':
    entry()

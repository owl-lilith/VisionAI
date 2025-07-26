part of 'image_editor_bloc.dart';

abstract class ImageEditorState {}

class ImageEditorInitial extends ImageEditorState {}

class ImageEditorLoading extends ImageEditorState {}

class ImageEditorImageSelected extends ImageEditorState {
  final File imageFile;
  ImageEditorImageSelected(this.imageFile);
}

class ImageEditorResult extends ImageEditorState {
  final File originalImage;
  final File editedImage;
  final List<File>? paletteImages;
  final File? filterImage;
  final File? skinMaskImage;

  ImageEditorResult({
    required this.originalImage,
    required this.editedImage,
    this.paletteImages,
    this.filterImage,
    this.skinMaskImage,
  });
}

class ImageEditorError extends ImageEditorState {
  final String message;
  ImageEditorError(this.message);
}

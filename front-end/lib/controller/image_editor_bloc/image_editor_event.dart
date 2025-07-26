part of 'image_editor_bloc.dart';


abstract class ImageEditorEvent {}

class UploadImageEvent extends ImageEditorEvent {
  final File imageFile;
  UploadImageEvent(this.imageFile);
}

class ApplyColorBalancingEvent extends ImageEditorEvent {
  final File imageFile;
  final int paletteSize;
  final double filterDegree;
  final int blendDegree;
  ApplyColorBalancingEvent({
    required this.imageFile,
    required this.paletteSize,
    required this.filterDegree,
    required this.blendDegree,
  });
}

class ApplyDreamyFilterEvent extends ImageEditorEvent {
  final File imageFile;
  final double brightness;
  final int maxDimmed;
  final int highlightSize;
  ApplyDreamyFilterEvent({
    required this.imageFile,
    required this.brightness,
    required this.maxDimmed,
    required this.highlightSize,
  });
}

class ApplyPureSkinEvent extends ImageEditorEvent {
  final File imageFile;
  final double blend;
  final int pure;
  ApplyPureSkinEvent({
    required this.imageFile,
    required this.blend,
    required this.pure,
  });
}

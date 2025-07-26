import 'dart:io';

import 'package:bloc/bloc.dart';
import 'package:equatable/equatable.dart';

part 'image_editor_event.dart';
part 'image_editor_state.dart';

class ImageEditorBloc extends Bloc<ImageEditorEvent, ImageEditorState> {
  ImageEditorBloc() : super(ImageEditorInitial()) {
    on<UploadImageEvent>(_onUploadImage);
    on<ApplyColorBalancingEvent>(_onApplyColorBalancing);
    on<ApplyDreamyFilterEvent>(_onApplyDreamyFilter);
    on<ApplyPureSkinEvent>(_onApplyPureSkin);
  }

  Future<void> _onUploadImage(
      UploadImageEvent event, Emitter<ImageEditorState> emit) async {
    emit(ImageEditorLoading());
    try {
      emit(ImageEditorImageSelected(event.imageFile));
    } catch (e) {
      emit(ImageEditorError(e.toString()));
    }
  }

  Future<void> _onApplyColorBalancing(
      ApplyColorBalancingEvent event, Emitter<ImageEditorState> emit) async {
    emit(ImageEditorLoading());
    try {
      // TODO: Implement API call to FastAPI endpoint
      // final result = await apiClient.applyColorBalancing(...);
      await Future.delayed(const Duration(seconds: 2)); // Mock delay
      emit(ImageEditorResult(
        originalImage: event.imageFile,
        editedImage: event.imageFile, // Replace with actual result
        paletteImages: [], // Replace with actual result
        filterImage: event.imageFile, // Replace with actual result
      ));
    } catch (e) {
      emit(ImageEditorError(e.toString()));
    }
  }

  Future<void> _onApplyDreamyFilter(
      ApplyDreamyFilterEvent event, Emitter<ImageEditorState> emit) async {
    emit(ImageEditorLoading());
    try {
      // TODO: Implement API call to FastAPI endpoint
      // final result = await apiClient.applyDreamyFilter(...);
      await Future.delayed(const Duration(seconds: 2)); // Mock delay
      emit(ImageEditorResult(
        originalImage: event.imageFile,
        editedImage: event.imageFile, // Replace with actual result
      ));
    } catch (e) {
      emit(ImageEditorError(e.toString()));
    }
  }

  Future<void> _onApplyPureSkin(
      ApplyPureSkinEvent event, Emitter<ImageEditorState> emit) async {
    emit(ImageEditorLoading());
    try {
      // TODO: Implement API call to FastAPI endpoint
      // final result = await apiClient.applyPureSkin(...);
      await Future.delayed(const Duration(seconds: 2)); // Mock delay
      emit(ImageEditorResult(
        originalImage: event.imageFile,
        editedImage: event.imageFile, // Replace with actual result
        skinMaskImage: event.imageFile, // Replace with actual result
      ));
    } catch (e) {
      emit(ImageEditorError(e.toString()));
    }
  }
}

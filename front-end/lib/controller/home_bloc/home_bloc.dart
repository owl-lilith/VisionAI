import 'dart:io';
import 'dart:convert';

import 'package:bloc/bloc.dart';
import 'package:equatable/equatable.dart';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';
import 'package:image_picker/image_picker.dart';
import 'package:path/path.dart';

part 'home_event.dart';
part 'home_state.dart';

File? imageFile;
String? searchText;

class HomeBloc extends Bloc<HomeEvent, HomeState> {
  HomeBloc() : super(HomeInitial()) {
    on<PerformSearch>(_onPerformSearch);
    on<PickImage>(_onPickImage);
    on<ThinkDeeper>(_onThinkDeeper);
  }

  Future<void> _onPickImage(PickImage event, Emitter<HomeState> emit) async {
    try {
      emit(PickImageLoading());
      final picker = ImagePicker();
      final pickedFile = await picker.pickImage(source: ImageSource.gallery);
      if (pickedFile != null) {
        imageFile = File(pickedFile.path);
      }
      emit(PickImageSuccess(imageFile!));
    } catch (e) {
      print(e.toString());
      emit(SearchFailure(e.toString()));
    }
  }

  Future<void> _onPerformSearch(
    PerformSearch event,
    Emitter<HomeState> emit,
  ) async {
    emit(SearchLoading());

    try {
      final uri = Uri.parse('http://192.168.59.1:8000/search');
      var request = http.MultipartRequest('POST', uri);

      if (event.searchText != null && event.searchText!.isNotEmpty) {
        request.fields['text'] = event.searchText!;
      }

      if (event.imageFile != null) {
        final file = event.imageFile!;
        final stream = http.ByteStream(file.openRead());
        final length = await file.length();

        var multipartFile = http.MultipartFile(
          'photo',
          stream,
          length,
          filename: basename(file.path),
          contentType: MediaType('image', 'jpeg'),
        );

        request.files.add(multipartFile);
      }
      // print('request');
      // print(request);

      final response = await request.send();

      final responseBody = await response.stream.bytesToString();

      if (response.statusCode == 200) {
        final responseData = json.decode(responseBody);
        emit(SearchSuccess(responseData));
        emit(
          NavigateToResults(
            searchText: event.searchText,
            imageFile: event.imageFile,
            results: responseData,
          ),
        );
      } else {
        emit(SearchFailure('Server error: ${response.statusCode}'));
      }
    } catch (e) {
      print(e.toString());
      emit(SearchFailure(e.toString()));
    }
    // emit(
    //   NavigateToResults(
    //     searchText: event.searchText,
    //     imageFile: event.imageFile,
    //     results: {"output": []},
    //   ),
    // );
  }

  Future<void> _onThinkDeeper(
    ThinkDeeper event,
    Emitter<HomeState> emit,
  ) async {
    emit(ThinkDeeperLoading());

    try {
      final uri = Uri.parse('http://192.168.59.1:8000/think_deeper');
      var request = http.MultipartRequest('POST', uri);

      if (event.searchText != null && event.searchText!.isNotEmpty) {
        request.fields['text'] = event.searchText!;
      }

      if (event.imageFile != null) {
        final file = event.imageFile!;
        final stream = http.ByteStream(file.openRead());
        final length = await file.length();

        var multipartFile = http.MultipartFile(
          'photo',
          stream,
          length,
          filename: basename(file.path),
          contentType: MediaType('image', 'jpeg'),
        );

        request.files.add(multipartFile);
      }

      final response = await request.send();

      final responseBody = await response.stream.bytesToString();

      if (response.statusCode == 200) {
        final responseData = json.decode(responseBody);
        emit(ThinkDeeperSuccess(responseData));
      } else {
        emit(ThinkDeeperFailure('Server error: ${response.statusCode}'));
      }
    } catch (e) {
      print(e.toString());
      emit(ThinkDeeperFailure(e.toString()));
    }
  }
}

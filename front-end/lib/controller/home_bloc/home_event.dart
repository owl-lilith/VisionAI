part of 'home_bloc.dart';

abstract class HomeEvent extends Equatable {
  const HomeEvent();

  @override
  List<Object> get props => [];
}

class PickImage extends HomeEvent {
  const PickImage();
}

class PerformSearch extends HomeEvent {
  final String? searchText;
  final File? imageFile;

  const PerformSearch({this.searchText, this.imageFile});

  @override
  List<Object> get props => [searchText ?? '', imageFile ?? ''];
}

class ThinkDeeper extends HomeEvent {
  final String? searchText;
  final File? imageFile;

  const ThinkDeeper({this.searchText, this.imageFile});

  @override
  List<Object> get props => [searchText ?? '', imageFile ?? ''];
}

// Add this to your HomeState
class NavigateToResults extends HomeState {
  final String? searchText;
  final File? imageFile;
  final Map<String, dynamic> results;

  const NavigateToResults({
    required this.searchText,
    required this.imageFile,
    required this.results,
  });

  @override
  List<Object> get props => [searchText ?? '', imageFile?.path ?? '', results];
}

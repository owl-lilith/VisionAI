part of 'home_bloc.dart';

abstract class HomeState extends Equatable {
  const HomeState();

  @override
  List<Object> get props => [];
}

final class HomeInitial extends HomeState {}

class PickImageLoading extends HomeState {}

class PickImageSuccess extends HomeState {
  final File imageFile;

  const PickImageSuccess(this.imageFile);

  @override
  List<Object> get props => [imageFile];
}

class PickImageFailure extends HomeState {
  final String error;

  const PickImageFailure(this.error);

  @override
  List<Object> get props => [error];
}

class SearchLoading extends HomeState {}

class SearchSuccess extends HomeState {
  final dynamic results;

  const SearchSuccess(this.results);

  @override
  List<Object> get props => [results];
}

class SearchFailure extends HomeState {
  final String error;

  const SearchFailure(this.error);

  @override
  List<Object> get props => [error];
}

class ThinkDeeperLoading extends HomeState {}

class ThinkDeeperSuccess extends HomeState {
  final dynamic results;

  const ThinkDeeperSuccess(this.results);

  @override
  List<Object> get props => [results];
}

class ThinkDeeperFailure extends HomeState {
  final String error;

  const ThinkDeeperFailure(this.error);

  @override
  List<Object> get props => [error];
}

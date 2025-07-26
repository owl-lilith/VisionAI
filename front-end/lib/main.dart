import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:image_search_engine_front_end/controller/image_editor_bloc/image_editor_bloc.dart';
import 'package:image_search_engine_front_end/presentation/screen/template_screen.dart';

import 'controller/home_bloc/home_bloc.dart';
import 'data/theme/theme_bloc.dart';
import 'data/theme/theme_metadata.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MultiBlocProvider(
      providers: [
        BlocProvider(create: (context) => HomeBloc()),
        BlocProvider(create: (context) => ThemeBloc()),
        BlocProvider(create: (context) => ImageEditorBloc()),

        // Add other BLoCs here as needed
      ],
      child: BlocBuilder<ThemeBloc, ThemeMode>(
        builder: (context, state) {
          return MaterialApp(
            title: 'Image Search Engine',

            debugShowCheckedModeBanner: false,
            theme: lightTheme,
            darkTheme: darkTheme,
            themeMode: state,
            home: BlocBuilder<HomeBloc, HomeState>(
              builder: (context, state) {
                // You can add global state handling here if needed
                return TemplateScreen();
              },
            ),
          );
        },
      ),
    );
  }
}

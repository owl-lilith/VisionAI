import 'dart:convert';
import 'dart:io';
import 'dart:ui';

import 'package:flutter/material.dart';

const String filtersOptionsDirectory =
    "D:\\image_search_engine_ai-end\\sources\\metadata\\home_page_filters_option.json";

Future<Map<String, dynamic>> readJsonFile() async {
  final file = File(filtersOptionsDirectory);
  final jsonString = await file.readAsString();
  return jsonDecode(jsonString);
}

class FiltersOptions {
  final List<String> colorsKeys;
  final Map<String, dynamic> colors;
  final List<String> facesKeys;
  final Map<String, dynamic> faces;
  final List<String> backgroundsKeys;
  final Map<String, dynamic> backgrounds;
  final List<String> objectsKeys;
  final Map<String, dynamic> objects;

  FiltersOptions({
    required this.colorsKeys,
    required this.colors,
    required this.facesKeys,
    required this.faces,
    required this.backgroundsKeys,
    required this.backgrounds,
    required this.objectsKeys,
    required this.objects,
  });

  factory FiltersOptions.fromJson(Map<String, dynamic> json) {
    return FiltersOptions(
      colorsKeys: List<String>.from(json['colors_keys'] ?? []),
      colors: Map<String, dynamic>.from(json['colors'] ?? {}),
      facesKeys: List<String>.from(json['faces_keys'] ?? []),
      faces: Map<String, dynamic>.from(json['faces'] ?? {}),
      backgroundsKeys: List<String>.from(json['backgrounds_keys'] ?? []),
      backgrounds: Map<String, dynamic>.from(json['backgrounds'] ?? {}),
      objectsKeys: List<String>.from(json['objects_keys'] ?? []),
      objects: Map<String, dynamic>.from(json['objects'] ?? {}),
    );
  }
}

Future<FiltersOptions> getFiltersOptions(Map<String, dynamic> json) async {
  return FiltersOptions.fromJson(json);
}

Color? getColorFromName(String name) {
  const colorMap = {
    'white': Colors.white,
    'black': Colors.black,
    'red': Colors.red,
    'green': Colors.green,
    'blue': Colors.blue,
    'orange': Colors.orange,
    'yellow': Colors.yellow,
    'purple': Colors.purple,
    'pink': Colors.pink,
    'brown': Colors.brown,
    'gray': Colors.grey,
    'grey': Colors.grey,
    'maroon': Color(0xFF800000),
    'turquoise': Color(0xFF40E0D0),
    'cyan': Colors.cyan,
    'navy blue': Color(0xFF000080),
    'navy': Color(0xFF000080),
    'gold': Color(0xFFFFD700),
    'silver': Color(0xFFC0C0C0),
    'beige': Color(0xFFF5F5DC),
    'teal': Colors.teal,
    'lime': Colors.lime,
    'olive': Color(0xFF808000),
    'violet': Color(0xFFEE82EE),
    'magenta': Colors.purpleAccent,
    'indigo': Colors.indigo,
  };

  return colorMap[name.toLowerCase()];
}

String? getNameFromColor(Color color) {
  Map<Color, String> colorMap = {
    Colors.white: 'white',
    Colors.black: 'black',
    Colors.red: 'red',
    Colors.green: 'green',
    Colors.blue: 'blue',
    Colors.orange: 'orange',
    Colors.yellow: 'yellow',
    Colors.purple: 'purple',
    Colors.pink: 'pink',
    Colors.brown: 'brown',
    Colors.grey: 'gray',
    Colors.grey: 'grey',
    Color(0xFF800000): 'maroon',
    Color(0xFF40E0D0): 'turquoise',
    Colors.cyan: 'cyan',
    Color(0xFF000080): 'navy blue',
    Color(0xFF000080): 'navy',
    Color(0xFFFFD700): 'gold',
    Color(0xFFC0C0C0): 'silver',
    Color(0xFFF5F5DC): 'beige',
    Colors.teal: 'teal',
    Colors.lime: 'lime',
    Color(0xFF808000): 'olive',
    Color(0xFFEE82EE): 'violet',
    Colors.purpleAccent: 'magenta',
    Colors.indigo: 'indigo',
  };

  return colorMap[color];
}

String? getFacesPath(String faceName) {
  Map<String, String> facesPaths = {
    'robin': "D:\\image_search_engine_ai-end\\sources\\faces\\robin\\000707_face0.jpg",
    'ted': "D:\\image_search_engine_ai-end\\sources\\faces\\ted\\000919.jpg",
    'marshel': "D:\\image_search_engine_ai-end\\sources\\faces\\marshel\\001010.jpg",
    'lily': "D:\\image_search_engine_ai-end\\sources\\faces\\lily\\000919 (3).jpg",
    'barney': "D:\\image_search_engine_ai-end\\sources\\faces\\barney\\000985 (2).jpg"
  };

  return facesPaths[faceName];
}

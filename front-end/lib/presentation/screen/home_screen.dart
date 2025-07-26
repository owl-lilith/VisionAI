import 'dart:io';
import 'dart:ui';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:image_search_engine_front_end/presentation/widget/background_home_image.dart';
import 'package:image_search_engine_front_end/presentation/widget/background_options_widget.dart';
import '../../controller/home_bloc/home_bloc.dart';
import '../../data/models/home_screen_filters_options.dart';
import '../../data/models/images_diractory.dart';
import 'package:flutter_bloc/flutter_bloc.dart';

import '../../data/theme/theme_metadata.dart';
import '../../data/models/home_screen_filters_options.dart';
import '../widget/shared_axis_page_route.dart';
import 'searching_result_screen.dart';

class HomeScreen extends StatefulWidget {
  static String routeName = '/home-screen';
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  FiltersOptions? filters;

  List<String> bg_images = getHomeImagesBackground();
  int _selectedTabIndex = -1;
  Color _selectedColor = Colors.blue;
  int _selectedFaceIndex = 0;
  List<String> _selectedObjects = ["cat"];
  List<String> _selectedBackgrounds = ["trench coat"];
  final TextEditingController _searchController = TextEditingController();

  List<String> _filteredOptions = [];

  List<Color> colorKeys = [];
  List<String> backgroundsKeys = [];
  List<String> facesKeys = [];
  List<String> objectsKeys = [];

  Map<String, dynamic> colors = {};
  Map<String, dynamic> backgrounds = {};
  Map<String, dynamic> faces = {};
  Map<String, dynamic> objects = {};
  @override
  void initState() {
    super.initState();
    loadFiltersData();
  }

  Future<void> loadFiltersData() async {
    final jsonData = await readJsonFile();
    filters = await getFiltersOptions(jsonData);
    colorKeys =
        filters!.colorsKeys.map((colorName) {
          return getColorFromName(colorName)!;
        }).toList();
    backgroundsKeys = filters!.backgroundsKeys;
    facesKeys = filters!.facesKeys;
    objectsKeys = filters!.objectsKeys;

    colors = filters!.colors;
    backgrounds = filters!.backgrounds;
    faces = filters!.faces;
    objects = filters!.objects;
  }

  void _showAllBackgroundsDialog() {
    _searchController.clear();
    _filteredOptions = backgroundsKeys;

    showDialog(
      context: context,
      builder:
          (context) => StatefulBuilder(
            builder: (context, setState) {
              return AlertDialog(
                title: const Text('Select Backgrounds'),
                content: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    TextField(
                      controller: _searchController,
                      decoration: const InputDecoration(
                        labelText: 'Search',
                        prefixIcon: Icon(Icons.search),
                      ),
                      onChanged: (value) {
                        setState(() {
                          _filteredOptions =
                              filters!.backgroundsKeys
                                  .where(
                                    (option) => option.toLowerCase().contains(
                                      value.toLowerCase(),
                                    ),
                                  )
                                  .toList();
                        });
                      },
                    ),
                    const SizedBox(height: 16),
                    SizedBox(
                      height: 300,
                      width: double.maxFinite,
                      child: ListView.builder(
                        itemCount: _filteredOptions.length,
                        itemBuilder: (context, index) {
                          final option = _filteredOptions[index];
                          return CheckboxListTile(
                            title: Text(option),
                            value: _selectedBackgrounds.contains(option),
                            onChanged: (selected) {
                              setState(() {
                                if (selected == true) {
                                  _selectedBackgrounds.add(option);
                                } else {
                                  _selectedBackgrounds.remove(option);
                                }
                              });
                            },
                          );
                        },
                      ),
                    ),
                  ],
                ),
                actions: [
                  TextButton(
                    onPressed: () => Navigator.pop(context),
                    child: const Text('Cancel'),
                  ),
                  TextButton(
                    onPressed: () {
                      Navigator.pop(context);
                      setState(() {});
                    },
                    child: const Text('Continue'),
                  ),
                ],
              );
            },
          ),
    );
  }

  @override
  Widget build(BuildContext context) {
    Size size = MediaQuery.of(context).size;

    return Scaffold(
      body: SizedBox(
        width: size.width,
        height: size.height,
        child: Row(
          children: [
            // Sidebar
            SizedBox(
              width: size.width * 0.2,
              child: Column(
                children: [
                  const SizedBox(height: 20),
                  const Text(
                    'Mach Your Images By',
                    style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(height: 20),
                  _buildSidebarTabs(),
                ],
              ),
            ),

            // Main content
            Expanded(
              flex: 3,
              child: Column(
                children: [
                  // Search Bar
                  Container(
                    width: size.width * 0.5,
                    padding: const EdgeInsets.all(16.0),
                    child: Hero(
                      tag: 'search-box',
                      child: Material(
                        type: MaterialType.transparency,
                        child: Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            Expanded(
                              flex: 1,
                              child: BlocConsumer<HomeBloc, HomeState>(
                                listener: (context, state) {
                                  if (state is SearchFailure) {
                                    ScaffoldMessenger.of(context).showSnackBar(
                                      SnackBar(content: Text(state.error)),
                                    );
                                  } else if (state is NavigateToResults) {
                                    Navigator.of(context).pushReplacement(
                                      SharedAxisPageRoute(
                                        page: SearchResultsScreen(
                                          results: state.results,
                                        ),
                                      ),
                                    );
                                  }
                                },
                                builder: (context, state) {
                                  if (state is SearchLoading) {
                                    return const Center(
                                      child: CircularProgressIndicator(),
                                    );
                                  }
                                  return IconButton(
                                    onPressed: () {
                                      context.read<HomeBloc>().add(
                                        PerformSearch(
                                          searchText: searchText,
                                          imageFile: imageFile,
                                        ),
                                      );
                                    },
                                    icon: Icon(Icons.search),
                                  );
                                },
                              ),
                            ),
                            Expanded(
                              flex: 4,
                              child: TextField(
                                onChanged:
                                    (text) => setState(() {
                                      searchText = text;
                                    }),
                                decoration: InputDecoration(
                                  hintText:
                                      'Search your images with thousands of prompts',
                                  border: OutlineInputBorder(
                                    borderRadius: BorderRadius.circular(30.0),
                                  ),
                                  filled: true,
                                  contentPadding: const EdgeInsets.symmetric(
                                    vertical: 15,
                                    horizontal: 10,
                                  ),
                                ),
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ),

                  Expanded(
                    flex: 7,
                    child: SizedBox(
                      height: size.height * 2,
                      child: ListView(
                        children: [
                          // Gradient Container with Blur Effect
                          Container(
                            margin: const EdgeInsets.all(20),
                            padding: const EdgeInsets.all(20),
                            width: MediaQuery.of(context).size.width * 0.7,
                            height: MediaQuery.of(context).size.height * 0.7,
                            decoration: BoxDecoration(
                              gradient: LinearGradient(
                                begin: Alignment.topLeft,
                                end: Alignment.bottomRight,
                                colors: [
                                  Colors.blue.shade400,
                                  Colors.purple.shade400,
                                ],
                              ),
                              borderRadius: BorderRadius.circular(12),
                            ),
                            child: Stack(
                              children: [
                                BackgroundHomeImage(
                                  bg_images: bg_images,
                                  bg_images_index: 0,
                                  alignment: Alignment.bottomLeft,
                                ),
                                BackgroundHomeImage(
                                  bg_images: bg_images,
                                  bg_images_index: 1,
                                  alignment: Alignment.bottomRight,
                                ),
                                BackgroundHomeImage(
                                  bg_images: bg_images,
                                  bg_images_index: 2,
                                  alignment: Alignment.topLeft,
                                ),
                                BackgroundHomeImage(
                                  bg_images: bg_images,
                                  bg_images_index: 3,
                                  alignment: Alignment.topRight,
                                ),

                                BlocConsumer<HomeBloc, HomeState>(
                                  listener: (context, state) {
                                    if (state is PickImageFailure) {
                                      ScaffoldMessenger.of(
                                        context,
                                      ).showSnackBar(
                                        SnackBar(content: Text(state.error)),
                                      );
                                    }
                                  },
                                  builder: (context, state) {
                                    if (state is PickImageLoading) {
                                      return const Center(
                                        child: CircularProgressIndicator(),
                                      );
                                    } else if (imageFile == null) {
                                      return Text('');
                                    }
                                    return Hero(
                                      tag: 'search-image',
                                      child: Material(
                                        type: MaterialType.transparency,
                                        child: SizedBox(
                                          width:
                                              MediaQuery.of(
                                                context,
                                              ).size.width *
                                              0.7,
                                          height:
                                              MediaQuery.of(
                                                context,
                                              ).size.height *
                                              0.7,
                                          child: ClipRRect(
                                            borderRadius: BorderRadius.circular(
                                              12,
                                            ),
                                            child: Center(
                                              child: Image.file(
                                                width:
                                                    MediaQuery.of(
                                                      context,
                                                    ).size.width *
                                                    0.7,
                                                height:
                                                    MediaQuery.of(
                                                      context,
                                                    ).size.height *
                                                    0.7,
                                                imageFile!,
                                                fit: BoxFit.cover,
                                              ),
                                            ),
                                          ),
                                        ),
                                      ),
                                    );
                                  },
                                ),

                                Center(
                                  child: Container(
                                    margin: EdgeInsets.all(20),
                                    height:
                                        MediaQuery.of(context).size.height *
                                        0.4,
                                    width:
                                        MediaQuery.of(context).size.width * 0.4,
                                    alignment: Alignment.center,
                                    child: Stack(
                                      children: [
                                        // Blurred Container
                                        Positioned.fill(
                                          child: ClipRRect(
                                            borderRadius: BorderRadius.circular(
                                              12,
                                            ),
                                            child: BackdropFilter(
                                              filter: ImageFilter.blur(
                                                sigmaX: 5,
                                                sigmaY: 5,
                                              ),
                                              child: Container(
                                                padding: const EdgeInsets.all(
                                                  16.0,
                                                ),
                                                child: Column(
                                                  mainAxisAlignment:
                                                      MainAxisAlignment.center,
                                                  children: [
                                                    Center(
                                                      child: const Text(
                                                        'Advance your image searching',
                                                        textAlign:
                                                            TextAlign.center,
                                                        style: TextStyle(
                                                          fontSize: 30,
                                                          fontWeight:
                                                              FontWeight.bold,
                                                        ),
                                                      ),
                                                    ),
                                                    const SizedBox(height: 8),
                                                    Center(
                                                      child: const Text(
                                                        'Find similar images by: context, nearly similar, same objects or distribution colors',
                                                        textAlign:
                                                            TextAlign.center,
                                                        style: TextStyle(
                                                          fontSize: 24,
                                                        ),
                                                      ),
                                                    ),

                                                    const SizedBox(height: 8),

                                                    Center(
                                                      child: ElevatedButton.icon(
                                                        onPressed: () {
                                                          context
                                                              .read<HomeBloc>()
                                                              .add(PickImage());
                                                        },
                                                        icon: const Icon(
                                                          Icons.upload,
                                                        ),
                                                        label: const Text(
                                                          'Upload your image',
                                                        ),
                                                        style: ElevatedButton.styleFrom(
                                                          foregroundColor:
                                                              Colors
                                                                  .blue
                                                                  .shade700,
                                                          shape: RoundedRectangleBorder(
                                                            borderRadius:
                                                                BorderRadius.circular(
                                                                  20,
                                                                ),
                                                          ),
                                                          padding:
                                                              const EdgeInsets.symmetric(
                                                                horizontal: 20,
                                                                vertical: 12,
                                                              ),
                                                        ),
                                                      ),
                                                    ),
                                                  ],
                                                ),
                                              ),
                                            ),
                                          ),
                                        ),
                                      ],
                                    ),
                                  ),
                                ),
                              ],
                            ),
                          ),

                          // Main Content All Images Area
                          Container(
                            margin: const EdgeInsets.all(20),
                            height: MediaQuery.of(context).size.height * 0.88,
                            child: _buildContentForSelectedTab(),
                          ),
                        ],
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSidebarTabs() {
    return Expanded(
      flex: 1,
      child: SingleChildScrollView(
        child: Column(
          children: [
            // Histogram Tab
            filtersTabWidget(
              index: 0,
              icon: Icons.bar_chart,
              title: 'Colors',
              content: Column(
                children: [
                  const SizedBox(height: 10),
                  // Color selection
                  const Text('Select Color:'),
                  const SizedBox(height: 8),
                  Wrap(
                    spacing: 8, // Horizontal space between items
                    runSpacing: 8, // Vertical space between lines
                    children:
                        colorKeys.map((color) {
                          final isSelected = _selectedColor == color;
                          return GestureDetector(
                            onTap: () {
                              setState(() {
                                _selectedColor = color;
                              });
                            },
                            child: Container(
                              width: 30,
                              height: 30,
                              decoration: BoxDecoration(
                                color: color,
                                shape: BoxShape.circle,
                                border:
                                    isSelected
                                        ? Border.all(
                                          color:
                                              Theme.of(
                                                context,
                                              ).colorScheme.primary,
                                          width: 2,
                                        )
                                        : null,
                                boxShadow:
                                    isSelected
                                        ? [
                                          BoxShadow(
                                            color: Colors.black.withOpacity(
                                              0.3,
                                            ),
                                            blurRadius: 4,
                                            spreadRadius: 1,
                                          ),
                                        ]
                                        : null,
                              ),
                              child:
                                  isSelected
                                      ? const Center(
                                        child: Icon(
                                          Icons.check,
                                          size: 14,
                                          color: Colors.white,
                                        ),
                                      )
                                      : null,
                            ),
                          );
                        }).toList(),
                  ),
                  const SizedBox(height: 15),
                ],
              ),
            ), // Background Tab
            filtersTabWidget(
              index: 1,
              icon: Icons.photo_library,
              title: 'Background',
              content: Builder(
                builder: (context) {
                  final initialBackgroundOptions =
                      backgroundsKeys.length > 10
                          ? backgroundsKeys.sublist(0, 10)
                          : backgroundsKeys;

                  return Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Wrap(
                        spacing: 8,
                        runSpacing: 8,
                        children:
                            initialBackgroundOptions.map((category) {
                              return ChoiceChip(
                                label: Text(category),
                                selected: _selectedBackgrounds.contains(
                                  category,
                                ),
                                onSelected: (selected) {
                                  setState(() {
                                    if (selected) {
                                      _selectedBackgrounds.add(category);
                                    } else {
                                      _selectedBackgrounds.remove(category);
                                    }
                                  });
                                },
                              );
                            }).toList(),
                      ),
                      if (backgroundsKeys.length > 10)
                        TextButton(
                          onPressed: _showAllBackgroundsDialog,
                          child: const Text('+ Show More'),
                        ),
                      // Show additional selected options
                      if (_selectedBackgrounds.any(
                        (bg) => !initialBackgroundOptions.contains(bg),
                      ))
                        Wrap(
                          spacing: 8,
                          runSpacing: 8,
                          children:
                              _selectedBackgrounds
                                  .where(
                                    (bg) =>
                                        !initialBackgroundOptions.contains(bg),
                                  )
                                  .map((category) {
                                    return ChoiceChip(
                                      label: Text(category),
                                      selected: true,
                                      onSelected: (selected) {
                                        setState(() {
                                          _selectedBackgrounds.remove(category);
                                        });
                                      },
                                    );
                                  })
                                  .toList(),
                        ),
                    ],
                  );
                },
              ),
            ),
            filtersTabWidget(
              index: 2,
              icon: Icons.face,
              title: 'Faces',
              content: Column(
                children: [
                  const SizedBox(height: 10),
                  const Text('Select Face:'),
                  const SizedBox(height: 8),
                  Wrap(
                    spacing: 8,
                    runSpacing: 8,
                    children:
                        facesKeys.asMap().entries.map((entry) {
                          final index = entry.key;
                          final face = entry.value;
                          final isSelected = _selectedFaceIndex == index;
                          return GestureDetector(
                            onTap: () {
                              setState(() {
                                _selectedFaceIndex = index;
                              });
                            },
                            child: Column(
                              children: [
                                Container(
                                  width: 60,
                                  height: 60,
                                  decoration: BoxDecoration(
                                    shape: BoxShape.circle,
                                    border:
                                        isSelected
                                            ? Border.all(
                                              color:
                                                  Theme.of(
                                                    context,
                                                  ).colorScheme.primary,
                                              width: 2,
                                            )
                                            : null,
                                    boxShadow:
                                        isSelected
                                            ? [
                                              BoxShadow(
                                                color: Colors.black.withOpacity(
                                                  0.3,
                                                ),
                                                blurRadius: 4,
                                                spreadRadius: 1,
                                              ),
                                            ]
                                            : null,
                                  ),
                                  child: ClipOval(
                                    child: Image.file(
                                      File(getFacesPath(face)!),
                                      fit: BoxFit.cover,
                                    ),
                                  ),
                                ),

                                const SizedBox(height: 4),
                                Text(
                                  face.split('/').last.replaceAll('.png', ''),
                                  style: TextStyle(
                                    color:
                                        isSelected
                                            ? Colors.blue
                                            : Theme.of(
                                              context,
                                            ).colorScheme.inverseSurface,
                                    fontWeight:
                                        isSelected
                                            ? FontWeight.bold
                                            : FontWeight.normal,
                                  ),
                                ),
                              ],
                            ),
                          );
                        }).toList(),
                  ),
                ],
              ),
            ),
            filtersTabWidget(
              index: 3,
              icon: Icons.category,
              title: 'Objects',
              content: Column(
                children: [
                  const SizedBox(height: 10),
                  const Text('Select Objects:'),
                  const SizedBox(height: 8),
                  Wrap(
                    spacing: 8,
                    runSpacing: 8,
                    children:
                        objectsKeys.map((object) {
                          final isSelected = _selectedObjects.contains(object);
                          return FilterChip(
                            label: Text(object),
                            selected: isSelected,
                            onSelected: (selected) {
                              setState(() {
                                if (selected) {
                                  _selectedObjects.add(object);
                                } else {
                                  _selectedObjects.remove(object);
                                }
                              });
                            },
                            selectedColor: Colors.blue.shade100,
                            checkmarkColor: Colors.blue,
                            labelStyle: TextStyle(
                              color:
                                  isSelected
                                      ? Colors.blue
                                      : Theme.of(
                                        context,
                                      ).colorScheme.inverseSurface,
                            ),
                          );
                        }).toList(),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget filtersTabWidget({
    required int index,
    required IconData icon,
    required String title,
    required Widget content,
  }) {
    return Container(
      margin: const EdgeInsets.symmetric(vertical: 4, horizontal: 8),
      decoration: BoxDecoration(
        color:
            _selectedTabIndex == index
                ? Theme.of(context).secondaryHeaderColor
                : Theme.of(context).primaryColor,
        borderRadius: BorderRadius.circular(8),
        border: Border.all(
          color:
              _selectedTabIndex == index
                  ? Colors.blue
                  : Theme.of(context).secondaryHeaderColor,
        ),
      ),
      child: InkWell(
        onTap: () {
          setState(() {
            _selectedTabIndex = index;
          });
        },
        child: Padding(
          padding: const EdgeInsets.all(12.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Icon(icon, color: Colors.blue),
                  const SizedBox(width: 10),
                  Text(
                    title,
                    style: const TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ],
              ),
              if (_selectedTabIndex == index) content,
            ],
          ),
        ),
      ),
    );
  }

  String _getTabName(int index) {
    switch (index) {
      case 0:
        return 'Colors';
      case 1:
        return 'Background';
      case 2:
        return 'Faces';
      case 3:
        return 'Objects';
      default:
        return '';
    }
  }

  // Helper method to get color name
  Widget _buildContentForSelectedTab() {
    if (filters == null) {
      return const Center(child: CircularProgressIndicator());
    }

    switch (_selectedTabIndex) {
      case 0: // Colors
        return selectedTabContent(colors[getNameFromColor(_selectedColor)]);
      case 1: // Backgrounds
        return selectedTabContent(backgrounds[_selectedBackgrounds[0]]);
      case 2: // Faces
        return selectedTabContent(faces[facesKeys[_selectedFaceIndex]]);
      case 3: // Objects
        return selectedTabContent(objects[_selectedObjects[0]]);
      default:
        return Container();
    }
  }

  Widget selectedTabContent(List<dynamic> contentList) {
    return GridView.builder(
      physics: BouncingScrollPhysics(),
      gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
        crossAxisCount: 5,
        crossAxisSpacing: 8,
        mainAxisSpacing: 8,
        childAspectRatio: 1,
      ),
      itemCount: contentList.length,
      itemBuilder: (context, index) {
        return Card(
          elevation: 2,
          child: ClipRRect(
            borderRadius: BorderRadius.circular(12),
            child: Image.file(File(contentList[index]), fit: BoxFit.cover),
          ),
        );
      },
    );
  }
}

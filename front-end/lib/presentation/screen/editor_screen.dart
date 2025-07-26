import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:image_picker/image_picker.dart';

import '../../controller/image_editor_bloc/image_editor_bloc.dart';

class EditorScreen extends StatefulWidget {
  static String routeName = '/editor-screen';
  const EditorScreen({super.key});

  @override
  State<EditorScreen> createState() => _EditorScreenState();
}

class _EditorScreenState extends State<EditorScreen> {
  File? _selectedImage;
  int _activeFilterIndex = -1;

  final ImagePicker _picker = ImagePicker();
  final Map<String, dynamic> _colorBalancingParams = {
    'paletteSize': 5,
    'filterDegree': 0.5,
    'blendDegree': 5,
  };
  final Map<String, dynamic> _dreamyFilterParams = {
    'brightness': 0.5,
    'maxDimmed': 220,
    'highlightSize': 100,
  };
  final Map<String, dynamic> _pureSkinParams = {'blend': 0.5, 'pure': 7};

  Future<void> _pickImage() async {
    final XFile? image = await _picker.pickImage(source: ImageSource.gallery);
    if (image != null) {
      setState(() {
        _selectedImage = File(image.path);
        _activeFilterIndex = -1;
      });
      context.read<ImageEditorBloc>().add(UploadImageEvent(_selectedImage!));
    }
  }

  void _applyColorBalancing() {
    if (_selectedImage == null) return;
    context.read<ImageEditorBloc>().add(
      ApplyColorBalancingEvent(
        imageFile: _selectedImage!,
        paletteSize: _colorBalancingParams['paletteSize'],
        filterDegree: _colorBalancingParams['filterDegree'],
        blendDegree: _colorBalancingParams['blendDegree'],
      ),
    );
  }

  void _applyDreamyFilter() {
    if (_selectedImage == null) return;
    context.read<ImageEditorBloc>().add(
      ApplyDreamyFilterEvent(
        imageFile: _selectedImage!,
        brightness: _dreamyFilterParams['brightness'],
        maxDimmed: _dreamyFilterParams['maxDimmed'],
        highlightSize: _dreamyFilterParams['highlightSize'],
      ),
    );
  }

  void _applyPureSkin() {
    if (_selectedImage == null) return;
    context.read<ImageEditorBloc>().add(
      ApplyPureSkinEvent(
        imageFile: _selectedImage!,
        blend: _pureSkinParams['blend'],
        pure: _pureSkinParams['pure'],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Image Editor')),
      body: BlocConsumer<ImageEditorBloc, ImageEditorState>(
        listener: (context, state) {
          if (state is ImageEditorError) {
            ScaffoldMessenger.of(
              context,
            ).showSnackBar(SnackBar(content: Text(state.message)));
          }
        },
        builder: (context, state) {
          return SingleChildScrollView(
            padding: const EdgeInsets.all(16.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                // Image upload section
                _buildImageUploadSection(state),
                const SizedBox(height: 20),
                // Filter selection
                _buildFilterSelection(),
                const SizedBox(height: 20),
                // Active filter controls
                if (_activeFilterIndex != -1) _buildActiveFilterControls(),
                const SizedBox(height: 20),
                // Results display
                _buildResultsDisplay(state),
              ],
            ),
          );
        },
      ),
    );
  }

  Widget _buildImageUploadSection(ImageEditorState state) {
    return Padding(
      padding: const EdgeInsets.all(16.0),
      child: Column(
        children: [
          const Text(
            'Upload Image',
            style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 10),
          if (_selectedImage == null)
            ElevatedButton(
              onPressed: _pickImage,
              child: const Text('Select Image'),
            )
          else
            Column(
              children: [
                Image.file(_selectedImage!, height: 200, fit: BoxFit.contain),
                const SizedBox(height: 10),
                Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    ElevatedButton(
                      onPressed: _pickImage,
                      child: const Text('Change Image'),
                    ),
                    const SizedBox(width: 10),
                    ElevatedButton(
                      onPressed: () {
                        setState(() {
                          _selectedImage = null;
                          _activeFilterIndex = -1;
                        });
                      },
                      child: const Text('Remove'),
                    ),
                  ],
                ),
              ],
            ),
        ],
      ),
    );
  }

  Widget _buildFilterSelection() {
    return Padding(
      padding: const EdgeInsets.all(16.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Filters',
            style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 10),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: [
              FilterChip(
                label: const Text('Color Balancing'),
                selected: _activeFilterIndex == 0,
                onSelected: (selected) {
                  setState(() {
                    _activeFilterIndex = selected ? 0 : -1;
                  });
                },
              ),
              FilterChip(
                label: const Text('Dreamy Filter'),
                selected: _activeFilterIndex == 1,
                onSelected: (selected) {
                  setState(() {
                    _activeFilterIndex = selected ? 1 : -1;
                  });
                },
              ),
              FilterChip(
                label: const Text('Pure Skin'),
                selected: _activeFilterIndex == 2,
                onSelected: (selected) {
                  setState(() {
                    _activeFilterIndex = selected ? 2 : -1;
                  });
                },
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildActiveFilterControls() {
    return Padding(
      padding: const EdgeInsets.all(16.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            _activeFilterIndex == 0
                ? 'Color Balancing'
                : _activeFilterIndex == 1
                ? 'Dreamy Filter'
                : 'Pure Skin',
            style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 10),
          if (_activeFilterIndex == 0) ...[
            _buildSliderControl(
              label: 'Palette Size',
              value: _colorBalancingParams['paletteSize'].toDouble(),
              min: 2,
              max: 20,
              divisions: 18,
              onChanged: (value) {
                setState(() {
                  _colorBalancingParams['paletteSize'] = value.toInt();
                });
              },
            ),
            _buildSliderControl(
              label: 'Filter Degree',
              value: _colorBalancingParams['filterDegree'],
              min: 0,
              max: 1,
              divisions: 10,
              onChanged: (value) {
                setState(() {
                  _colorBalancingParams['filterDegree'] = value;
                });
              },
            ),
            _buildSliderControl(
              label: 'Blend Degree',
              value: _colorBalancingParams['blendDegree'].toDouble(),
              min: 1,
              max: 20,
              divisions: 19,
              onChanged: (value) {
                setState(() {
                  _colorBalancingParams['blendDegree'] = value.toInt();
                });
              },
            ),
            ElevatedButton(
              onPressed: _applyColorBalancing,
              child: const Text('Apply Color Balancing'),
            ),
          ] else if (_activeFilterIndex == 1) ...[
            _buildSliderControl(
              label: 'Brightness',
              value: _dreamyFilterParams['brightness'],
              min: 0,
              max: 1,
              divisions: 10,
              onChanged: (value) {
                setState(() {
                  _dreamyFilterParams['brightness'] = value;
                });
              },
            ),
            _buildSliderControl(
              label: 'Max Dimmed',
              value: _dreamyFilterParams['maxDimmed'].toDouble(),
              min: 0,
              max: 255,
              divisions: 255,
              onChanged: (value) {
                setState(() {
                  _dreamyFilterParams['maxDimmed'] = value.toInt();
                });
              },
            ),
            _buildSliderControl(
              label: 'Highlight Size',
              value: _dreamyFilterParams['highlightSize'].toDouble(),
              min: 1,
              max: 200,
              divisions: 199,
              onChanged: (value) {
                setState(() {
                  _dreamyFilterParams['highlightSize'] = value.toInt();
                });
              },
            ),
            ElevatedButton(
              onPressed: _applyDreamyFilter,
              child: const Text('Apply Dreamy Filter'),
            ),
          ] else if (_activeFilterIndex == 2) ...[
            _buildSliderControl(
              label: 'Blend',
              value: _pureSkinParams['blend'],
              min: 0,
              max: 1,
              divisions: 10,
              onChanged: (value) {
                setState(() {
                  _pureSkinParams['blend'] = value;
                });
              },
            ),
            _buildSliderControl(
              label: 'Pure',
              value: _pureSkinParams['pure'].toDouble(),
              min: 1,
              max: 20,
              divisions: 19,
              onChanged: (value) {
                setState(() {
                  _pureSkinParams['pure'] = value.toInt();
                });
              },
            ),
            ElevatedButton(
              onPressed: _applyPureSkin,
              child: const Text('Apply Pure Skin'),
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildSliderControl({
    required String label,
    required double value,
    required double min,
    required double max,
    required int divisions,
    required ValueChanged<double> onChanged,
  }) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text('$label: ${value.toStringAsFixed(2)}'),
        Slider(
          value: value,
          min: min,
          max: max,
          divisions: divisions,
          onChanged: onChanged,
        ),
      ],
    );
  }

  Widget _buildResultsDisplay(ImageEditorState state) {
    if (state is ImageEditorInitial || _selectedImage == null) {
      return const SizedBox();
    }

    if (state is ImageEditorLoading) {
      return const Center(child: CircularProgressIndicator());
    }

    if (state is ImageEditorResult) {
      return Card(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                'Results',
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 10),
              // Original and edited images
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: [
                  Column(
                    children: [
                      const Text('Original'),
                      const SizedBox(height: 5),
                      Image.file(
                        state.originalImage,
                        height: 150,
                        fit: BoxFit.contain,
                      ),
                    ],
                  ),
                  Column(
                    children: [
                      const Text('Edited'),
                      const SizedBox(height: 5),
                      Image.file(
                        state.editedImage,
                        height: 150,
                        fit: BoxFit.contain,
                      ),
                    ],
                  ),
                ],
              ),
              // Additional results for color balancing
              if (state.paletteImages != null &&
                  state.paletteImages!.isNotEmpty)
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const SizedBox(height: 20),
                    const Text(
                      'Color Palette',
                      style: TextStyle(fontWeight: FontWeight.bold),
                    ),
                    const SizedBox(height: 10),
                    SingleChildScrollView(
                      scrollDirection: Axis.horizontal,
                      child: Row(
                        children:
                            state.paletteImages!
                                .map(
                                  (image) => Padding(
                                    padding: const EdgeInsets.only(right: 8.0),
                                    child: Image.file(
                                      image,
                                      height: 50,
                                      width: 50,
                                      fit: BoxFit.cover,
                                    ),
                                  ),
                                )
                                .toList(),
                      ),
                    ),
                  ],
                ),
              // Filter image for color balancing
              if (state.filterImage != null)
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const SizedBox(height: 20),
                    const Text(
                      'Applied Filter',
                      style: TextStyle(fontWeight: FontWeight.bold),
                    ),
                    const SizedBox(height: 10),
                    Image.file(
                      state.filterImage!,
                      height: 150,
                      fit: BoxFit.contain,
                    ),
                  ],
                ),
              // Skin mask for pure skin
              if (state.skinMaskImage != null)
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const SizedBox(height: 20),
                    const Text(
                      'Skin Mask',
                      style: TextStyle(fontWeight: FontWeight.bold),
                    ),
                    const SizedBox(height: 10),
                    Image.file(
                      state.skinMaskImage!,
                      height: 150,
                      fit: BoxFit.contain,
                    ),
                  ],
                ),
            ],
          ),
        ),
      );
    }

    return const SizedBox();
  }
}

The main algorithms for seeding divide into: 1) spacepoint grouping and  2) seed finding.
### Spacepoint Grouping (only w/ CPU)

In spacepoint grouping, the spacepoint EDM are binned based on their positions in phi-axis and z-axis.
Main changes from ACTS are the followig:

- `grid.hpp` does not hold binned spcepoint objects (`m_values` of `Grid.hpp` in ACTS)) anymore. 
  Instead, the binned spacepoints are passed into the internal spacepoint container (`core/include/edm/internal_spacepoint.hpp`)

- `internal_spacepoint.hpp` EDM has the following header and item information:
  - header: the index of cuurent global bin and indices of neighborhood global bins which are found by `bin_finder.hpp`
  - item: position and variable

  The reason for recording the bin indices explicitly is to use the EDM directly inside the CUDA device kernel.

module mem #(parameter DW = 8, parameter N = 8) (
  input  logic clk,
  input logic rst_n,
  input  logic write_e,
  input  logic [$clog2(N*N)-1:0] waddr,
  input  logic [DW-1:0] wdata,
  input  logic [$clog2(N*N)-1:0] raddr,
  output logic [DW-1:0] rdata
);

  logic [DW-1:0] mem_array [0:N*N-1];

  // has to be on clock edge
  always_ff @(posedge clk) begin
      if (write_e)
          mem_array[waddr] <= wdata;
  end

  //instant read
  assign rdata = mem_array[raddr];

endmodule
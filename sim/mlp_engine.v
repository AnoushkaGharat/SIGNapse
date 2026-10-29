module mlp_engine #(
    parameter WDEPTH = 32768,
    parameter BDEPTH = 1024,
    parameter ADEPTH = 256,
    parameter DDEPTH = 64
) (
    input  wire              clk,
    input  wire              rst,
    input  wire              start,
    input  wire [7:0]        num_layers,
    input  wire              in_we,
    input  wire [15:0]       in_addr,
    input  wire signed [7:0] in_data,
    input  wire [15:0]       out_addr,
    output wire signed [7:0] out_data,
    output reg               done,
    output reg  [31:0]       cycles
);
    localparam S_IDLE = 3'd0, S_LOAD = 3'd1, S_MAC = 3'd2, S_POST = 3'd3, S_NEXT = 3'd4, S_DONE = 3'd5;

    reg signed [7:0]  wmem [0:WDEPTH-1];
    reg signed [31:0] bmem [0:BDEPTH-1];
    reg        [31:0] dmem [0:DDEPTH-1];
    reg signed [7:0]  act0 [0:ADEPTH-1];
    reg signed [7:0]  act1 [0:ADEPTH-1];

    reg [2:0]  state;
    reg [7:0]  layer;
    reg        sel;
    reg [15:0] k, n, i, j;
    reg [4:0]  shift;
    reg        relu;
    reg [31:0] wbase, bbase, wptr;

    wire signed [7:0]  a_in = sel ? act1[i] : act0[i];
    wire signed [7:0]  w_in = wmem[wptr];
    wire signed [31:0] acc;
    wire signed [7:0]  pp_out;

    mac_unit u_mac (
        .clk(clk), .rst(rst),
        .validity(state == S_MAC),
        .first_num(i == 16'd0),
        .a(a_in), .b(w_in),
        .acc(acc)
    );

    postproc u_pp (
        .acc(acc), .bias(bmem[bbase + j]),
        .shift(shift), .relu(relu),
        .out(pp_out)
    );

    assign out_data = sel ? act1[out_addr] : act0[out_addr];

    wire [31:0] d_k     = dmem[layer * 5 + 0];
    wire [31:0] d_n     = dmem[layer * 5 + 1];
    wire [31:0] d_cfg   = dmem[layer * 5 + 2];
    wire [31:0] d_wbase = dmem[layer * 5 + 3];
    wire [31:0] d_bbase = dmem[layer * 5 + 4];

    always @(posedge clk) begin
        if (in_we)
            act0[in_addr] <= in_data;
        if (rst) begin
            state  <= S_IDLE;
            done   <= 1'b0;
            sel    <= 1'b0;
            cycles <= 32'd0;
        end else begin
            if (state != S_IDLE && state != S_DONE)
                cycles <= cycles + 32'd1;
            case (state)
                S_IDLE, S_DONE: if (start) begin
                    done   <= 1'b0;
                    layer  <= 8'd0;
                    sel    <= 1'b0;
                    cycles <= 32'd0;
                    state  <= S_LOAD;
                end
                S_LOAD: begin
                    k     <= d_k[15:0];
                    n     <= d_n[15:0];
                    shift <= d_cfg[4:0];
                    relu  <= d_cfg[5];
                    wbase <= d_wbase;
                    bbase <= d_bbase;
                    wptr  <= d_wbase;
                    i     <= 16'd0;
                    j     <= 16'd0;
                    state <= S_MAC;
                end
                S_MAC: begin
                    if (i == k - 16'd1)
                        state <= S_POST;
                    else begin
                        i    <= i + 16'd1;
                        wptr <= wptr + n;
                    end
                end
                S_POST: begin
                    if (sel) act0[j] <= pp_out;
                    else     act1[j] <= pp_out;
                    if (j == n - 16'd1)
                        state <= S_NEXT;
                    else begin
                        j     <= j + 16'd1;
                        i     <= 16'd0;
                        wptr  <= wbase + j + 32'd1;
                        state <= S_MAC;
                    end
                end
                S_NEXT: begin
                    sel   <= ~sel;
                    layer <= layer + 8'd1;
                    if (layer + 8'd1 == num_layers) begin
                        done  <= 1'b1;
                        state <= S_DONE;
                    end else
                        state <= S_LOAD;
                end
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule

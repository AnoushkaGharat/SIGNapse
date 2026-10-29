module tb_mlp;
    reg clk = 1'b0;
    reg rst = 1'b1;
    reg start = 1'b0;
    reg in_we = 1'b0;
    reg [15:0] in_addr = 16'd0;
    reg signed [7:0] in_data = 8'sd0;
    reg [15:0] out_addr = 16'd0;
    reg [7:0] num_layers;
    wire signed [7:0] out_data;
    wire done;
    wire [31:0] cycles;

    always #5 clk = ~clk;

    mlp_engine dut (
        .clk(clk), .rst(rst), .start(start), .num_layers(num_layers),
        .in_we(in_we), .in_addr(in_addr), .in_data(in_data),
        .out_addr(out_addr), .out_data(out_data),
        .done(done), .cycles(cycles)
    );

    reg [8*512-1:0] wfile, bfile, dfile;
    integer nl, nin, nout, idx, v, r;

    initial begin
        if (!$value$plusargs("W=%s", wfile) || !$value$plusargs("B=%s", bfile) || !$value$plusargs("D=%s", dfile) ||
            !$value$plusargs("NL=%d", nl) || !$value$plusargs("NIN=%d", nin) || !$value$plusargs("NOUT=%d", nout)) begin
            $display("ERR missing plusargs");
            $finish;
        end
        $readmemh(wfile, dut.wmem);
        $readmemh(bfile, dut.bmem);
        $readmemh(dfile, dut.dmem);
        num_layers = nl;
        repeat (2) @(negedge clk);
        rst = 1'b0;
        $display("READY");
        $fflush;
        forever begin
            for (idx = 0; idx < nin; idx = idx + 1) begin
                r = $fscanf(32'h8000_0000, "%d", v);
                if (r != 1) $finish;
                @(negedge clk);
                in_we = 1'b1;
                in_addr = idx;
                in_data = v;
            end
            @(negedge clk);
            in_we = 1'b0;
            start = 1'b1;
            @(negedge clk);
            start = 1'b0;
            while (!done) @(negedge clk);
            $write("OUT");
            for (idx = 0; idx < nout; idx = idx + 1) begin
                out_addr = idx;
                #1;
                $write(" %0d", out_data);
            end
            $write(" %0d\n", cycles);
            $fflush;
        end
    end
endmodule

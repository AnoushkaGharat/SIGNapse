module matmul_top #(
    parameter N          = 8,
    parameter DW         = 8,
    parameter ACC_W      = 32,
    parameter MAXLAYER   = 4,
    parameter WMEM_AW    = 7,
    parameter BMEM_AW    = 4,
    parameter EN_A_REUSE = 1
)(
    input  logic clk,
    input  logic rst_n,

    input  logic start,
    output logic done,

    input  logic                    ld_en,
    input  logic                    ld_sel_ab,
    input  logic [9:0]              ld_addr,
    input  logic signed [ACC_W-1:0] ld_data,

    input  logic                    rd_en,
    input  logic [$clog2(N*N)-1:0]  rd_addr,
    output logic signed [ACC_W-1:0] rd_data
);

    localparam ADDR_W     = $clog2(N*N);
    localparam DIM_W      = $clog2(N+1);
    localparam LAY_W      = $clog2(MAXLAYER);
    localparam NUM_W      = $clog2(MAXLAYER+1);
    localparam CNT_W      = $clog2(N*N+1);
    localparam SH_W       = 5;
    localparam KIDX_W     = $clog2(N);
    localparam WB_LATENCY = 2;

    logic ld_act, ld_wgt, ld_bias, cfg_we;

    assign ld_act  = ld_en && !ld_sel_ab && (ld_addr[9:8] == 2'd0);
    assign ld_wgt  = ld_en && !ld_sel_ab && (ld_addr[9:8] == 2'd1);
    assign ld_bias = ld_en && !ld_sel_ab && (ld_addr[9:8] == 2'd2);
    assign cfg_we  = ld_en &&  ld_sel_ab;

    logic [DIM_W-1:0]   d_m   [0:MAXLAYER-1];
    logic [DIM_W-1:0]   d_k   [0:MAXLAYER-1];
    logic [DIM_W-1:0]   d_n   [0:MAXLAYER-1];
    logic [SH_W-1:0]    d_sh  [0:MAXLAYER-1];
    logic               d_ben [0:MAXLAYER-1];
    logic               d_ren [0:MAXLAYER-1];
    logic               d_pas [0:MAXLAYER-1];
    logic               d_tr  [0:MAXLAYER-1];
    logic [WMEM_AW-1:0] d_wb  [0:MAXLAYER-1];
    logic [BMEM_AW-1:0] d_bb  [0:MAXLAYER-1];

    logic [NUM_W-1:0] num_layers;
    logic [LAY_W-1:0] cfg_lay;

    assign cfg_lay = ld_addr[LAY_W:1];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            num_layers <= 0;
        end else if (cfg_we) begin
            if (ld_addr[3]) begin
                num_layers <= ld_data[NUM_W-1:0];
            end else if (!ld_addr[0]) begin
                d_m  [cfg_lay] <= ld_data[3:0];
                d_k  [cfg_lay] <= ld_data[7:4];
                d_n  [cfg_lay] <= ld_data[11:8];
                d_sh [cfg_lay] <= ld_data[16:12];
                d_ben[cfg_lay] <= ld_data[17];
                d_ren[cfg_lay] <= ld_data[18];
                d_pas[cfg_lay] <= ld_data[19];
                d_tr [cfg_lay] <= ld_data[20];
            end else begin
                d_wb [cfg_lay] <= ld_data[WMEM_AW-1:0];
                d_bb [cfg_lay] <= ld_data[8+BMEM_AW-1:8];
            end
        end
    end

    typedef enum logic [1:0] {
        IDLE,
        COMPUTE,
        FLUSH,
        DRAIN
    } state_t;

    state_t state;

    logic [LAY_W-1:0] layer;
    logic             cur;
    logic [2:0]       drain_cnt;

    logic [DIM_W-1:0]   m_r, k_r, n_r;
    logic [SH_W-1:0]    sh_r;
    logic               tr_r, ben_r, ren_r, pas_r;
    logic [WMEM_AW-1:0] wb_r;
    logic [BMEM_AW-1:0] bb_r;
    logic [CNT_W-1:0]   tot_r;

    logic [DIM_W-1:0]   i, j, k;

    logic [ADDR_W-1:0]  a_addr;
    logic [WMEM_AW-1:0] b_addr;

    logic mac_valid_in, mac_clear;

    logic [ADDR_W-1:0] addr_pipe  [0:2];
    logic [DIM_W-1:0]  col_pipe   [0:2];
    logic              final_pipe [0:2];

    logic [CNT_W-1:0] final_count;

    integer x;

    logic signed [DW-1:0]    a_rdata, a0_rdata, a1_rdata, b_rdata;
    logic signed [ACC_W-1:0] mac_out;
    logic                    mac_valid;

    assign a_rdata = cur ? a1_rdata : a0_rdata;

    logic signed [DW-1:0] a_row [0:N-1];
    logic [DIM_W-1:0]     a_k_q;
    logic                 a_hit_q;
    logic                 a_fill_q;
    logic signed [DW-1:0] a_operand;
    integer               ri;

    logic [KIDX_W-1:0] a_kidx;

    assign a_kidx    = a_k_q[KIDX_W-1:0];
    assign a_operand = a_hit_q ? a_row[a_kidx] : a_rdata;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (ri = 0; ri < N; ri = ri + 1)
                a_row[ri] <= '0;
        end else if (a_fill_q) begin
            a_row[a_kidx] <= a_rdata;
        end
    end

    mac_unit #(DW, ACC_W) mac (
        .clk       (clk),
        .rst_n     (rst_n),
        .a         (a_operand),
        .b         (b_rdata),
        .valid_in  (mac_valid_in),
        .clear_acc (mac_clear),
        .acc_out   (mac_out),
        .valid_out (mac_valid)
    );

    logic              c_we;
    logic [ADDR_W-1:0] c_addr;
    logic [DIM_W-1:0]  c_col;

    assign c_we   = mac_valid && final_pipe[2];
    assign c_addr = addr_pipe[2];
    assign c_col  = col_pipe[2];

    logic signed [ACC_W-1:0] bias_rdata, res32;
    logic signed [7:0]       act8;

    logic signed [ACC_W-1:0] acc_s1;
    logic signed [ACC_W-1:0] bias_s1;
    logic                    c_we_s1;
    logic [ADDR_W-1:0]       c_addr_s1;
    logic                    cur_s1;
    logic [SH_W-1:0]         sh_s1;
    logic                    ben_s1;
    logic                    ren_s1;
    logic                    pas_s1;

    logic                    c_we_s2;
    logic [ADDR_W-1:0]       c_addr_s2;
    logic                    cur_s2;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc_s1    <= '0;
            bias_s1   <= '0;
            c_we_s1   <= 1'b0;
            c_addr_s1 <= '0;
            cur_s1    <= 1'b0;
            sh_s1     <= '0;
            ben_s1    <= 1'b0;
            ren_s1    <= 1'b0;
            pas_s1    <= 1'b0;

            c_we_s2   <= 1'b0;
            c_addr_s2 <= '0;
            cur_s2    <= 1'b0;
        end else begin
            acc_s1    <= mac_out;
            bias_s1   <= bias_rdata;
            c_we_s1   <= c_we;
            c_addr_s1 <= c_addr;
            cur_s1    <= cur;
            sh_s1     <= sh_r;
            ben_s1    <= ben_r;
            ren_s1    <= ren_r;
            pas_s1    <= pas_r;

            c_we_s2   <= c_we_s1;
            c_addr_s2 <= c_addr_s1;
            cur_s2    <= cur_s1;
        end
    end

    postproc #(
        .ACC_W (ACC_W),
        .SH_W  (SH_W)
    ) post (
        .clk      (clk),
        .rst_n    (rst_n),
        .acc      (acc_s1),
        .bias     (bias_s1),
        .shift    (sh_s1),
        .bias_en  (ben_s1),
        .relu_en  (ren_s1),
        .passthru (pas_s1),
        .res32    (res32),
        .act8     (act8)
    );

    logic              a0_we, a1_we;
    logic [ADDR_W-1:0] a0_wa;
    logic [DW-1:0]     a0_wd;

    // idle host owns buffer 0
    // running each layer writes the buffer its not reading
    assign a0_we = (state == IDLE) ? ld_act : (c_we_s2 && cur_s2);
    assign a0_wa = (state == IDLE) ? ld_addr[ADDR_W-1:0] : c_addr_s2;
    assign a0_wd = (state == IDLE) ? ld_data[DW-1:0] : act8;
    assign a1_we = (state != IDLE) && c_we_s2 && !cur_s2;

    mem #(.DW(DW), .N(ADDR_W)) mem_A0 (
        .clk(clk), .rst_n(rst_n), .write_e(a0_we),
        .waddr(a0_wa), .wdata(a0_wd), .raddr(a_addr), .rdata(a0_rdata)
    );

    mem #(.DW(DW), .N(ADDR_W)) mem_A1 (
        .clk(clk), .rst_n(rst_n), .write_e(a1_we),
        .waddr(c_addr_s2), .wdata(act8), .raddr(a_addr), .rdata(a1_rdata)
    );

    logic [WMEM_AW-1:0] w_raddr;

    assign w_raddr = wb_r + b_addr;

    mem #(.DW(DW), .N(WMEM_AW)) mem_B (
        .clk(clk), .rst_n(rst_n), .write_e(ld_wgt),
        .waddr(ld_addr[5:0]), .wdata(ld_data[DW-1:0]),
        .raddr(w_raddr[5:0]), .rdata(b_rdata)
    );

    logic [BMEM_AW-1:0] bias_raddr;

    assign bias_raddr = bb_r + c_col[BMEM_AW-1:0];

    mem #(.DW(ACC_W), .N(BMEM_AW)) mem_BIAS (
        .clk(clk), .rst_n(rst_n), .write_e(ld_bias),
        .waddr(ld_addr[BMEM_AW-1:0]), .wdata(ld_data),
        .raddr(bias_raddr), .rdata(bias_rdata)
    );

    // result mem rewritten every layer
    // after done holds final layer 32 bit output
    logic signed [ACC_W-1:0] c_rdata;

    mem #(.DW(ACC_W), .N(ADDR_W)) mem_C (
        .clk(clk), .rst_n(rst_n), .write_e(c_we_s2),
        .waddr(c_addr_s2), .wdata(res32), .raddr(rd_addr), .rdata(c_rdata)
    );

    assign rd_data = c_rdata;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            done  <= 0;

            i <= 0;
            j <= 0;
            k <= 0;

            mac_valid_in <= 0;
            mac_clear    <= 0;

            final_count <= 0;
            layer       <= 0;
            cur         <= 0;
            drain_cnt   <= 0;

            m_r  <= 0; k_r   <= 0; n_r   <= 0; sh_r  <= 0;
            tr_r <= 0; ben_r <= 0; ren_r <= 0; pas_r <= 0;
            wb_r <= 0; bb_r  <= 0; tot_r <= 0;

            a_addr <= 0;
            b_addr <= 0;

            a_k_q    <= 0;
            a_hit_q  <= 0;
            a_fill_q <= 0;

            for (x = 0; x < 3; x = x + 1) begin
                addr_pipe[x]  <= 0;
                col_pipe[x]   <= 0;
                final_pipe[x] <= 0;
            end
        end else begin
            done         <= 0;
            mac_valid_in <= 0;
            mac_clear    <= 0;

            a_hit_q  <= 0;
            a_fill_q <= 0;

            addr_pipe[1]  <= addr_pipe[0];
            addr_pipe[2]  <= addr_pipe[1];
            col_pipe[1]   <= col_pipe[0];
            col_pipe[2]   <= col_pipe[1];
            final_pipe[1] <= final_pipe[0];
            final_pipe[2] <= final_pipe[1];

            if (c_we) begin
                if (final_count == tot_r - 1) begin
                    final_count <= 0;
                    drain_cnt   <= 0;
                    state       <= DRAIN;
                end else begin
                    final_count <= final_count + 1;
                end
            end

            case (state)

                IDLE: begin
                    if (start && num_layers != 0) begin
                        layer <= 0;
                        cur   <= 0;

                        m_r   <= d_m[0];
                        k_r   <= d_k[0];
                        n_r   <= d_n[0];
                        sh_r  <= d_sh[0];
                        tr_r  <= d_tr[0];
                        ben_r <= d_ben[0];
                        ren_r <= d_ren[0];
                        pas_r <= d_pas[0];
                        wb_r  <= d_wb[0];
                        bb_r  <= d_bb[0];
                        tot_r <= d_m[0] * d_n[0];

                        i <= 0;
                        j <= 0;
                        k <= 0;

                        final_count <= 0;
                        state       <= COMPUTE;
                    end
                end

                COMPUTE: begin
                    mac_valid_in <= 1;
                    mac_clear    <= (k == 0);

                    if (!EN_A_REUSE || (j == 0)) begin
                        a_addr <= tr_r ? (k * m_r + 6'(i)) : (i * k_r + 6'(k));
                    end

                    a_k_q    <= k;
                    a_fill_q <= EN_A_REUSE && (j == 0);
                    a_hit_q  <= EN_A_REUSE && (j != 0);

                    b_addr <= k * n_r + 7'(j);

                    addr_pipe[0]  <= i * n_r + 6'(j);
                    col_pipe[0]   <= j;
                    final_pipe[0] <= (k == k_r - 1);

                    if (k == k_r - 1) begin
                        k <= 0;
                        if (j == n_r - 1) begin
                            j <= 0;
                            if (i == m_r - 1) begin
                                i     <= 0;
                                state <= FLUSH;
                            end else begin
                                i <= i + 1;
                            end
                        end else begin
                            j <= j + 1;
                        end
                    end else begin
                        k <= k + 1;
                    end
                end

                FLUSH: begin
                    mac_valid_in  <= 0;
                    final_pipe[0] <= 0;
                end

                DRAIN: begin
                    mac_valid_in  <= 0;
                    final_pipe[0] <= 0;
                    drain_cnt     <= drain_cnt + 1;

                    if (drain_cnt == (2 + WB_LATENCY)) begin
                        if (3'(layer) == (num_layers - 1'b1)) begin
                            done  <= 1;
                            state <= IDLE;
                        end else begin
                            layer <= layer + 1;
                            cur   <= ~cur;

                            m_r   <= d_m[layer + 1];
                            k_r   <= d_k[layer + 1];
                            n_r   <= d_n[layer + 1];
                            sh_r  <= d_sh[layer + 1];
                            tr_r  <= d_tr[layer + 1];
                            ben_r <= d_ben[layer + 1];
                            ren_r <= d_ren[layer + 1];
                            pas_r <= d_pas[layer + 1];
                            wb_r  <= d_wb[layer + 1];
                            bb_r  <= d_bb[layer + 1];
                            tot_r <= d_m[layer + 1] * d_n[layer + 1];

                            i <= 0;
                            j <= 0;
                            k <= 0;

                            final_count <= 0;
                            state       <= COMPUTE;
                        end
                    end
                end

                default: begin
                    state <= IDLE;
                end

            endcase
        end
    end

endmodule


module mac_unit #(
    parameter DW    = 8,
    parameter ACC_W = 32
)(
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic signed [DW-1:0]    a,
    input  logic signed [DW-1:0]    b,
    input  logic                    valid_in,
    input  logic                    clear_acc,

    output logic signed [ACC_W-1:0] acc_out,
    output logic                    valid_out
);

    logic signed [2*DW-1:0] product_s1;
    logic                   valid_s1;
    logic                   first_term_s1;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            product_s1    <= 0;
            valid_s1      <= 0;
            first_term_s1 <= 0;
        end else begin
            product_s1    <= a * b;
            valid_s1      <= valid_in;
            first_term_s1 <= clear_acc;
        end
    end

    logic signed [ACC_W-1:0] acc;
    logic                    valid_s2;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc      <= 0;
            valid_s2 <= 0;
        end else begin
            valid_s2 <= valid_s1;
            if (valid_s1) begin
                if (first_term_s1)
                    acc <= {{(ACC_W-2*DW){product_s1[2*DW-1]}}, product_s1};
                else
                    acc <= acc + {{(ACC_W-2*DW){product_s1[2*DW-1]}}, product_s1};
            end
        end
    end

    assign acc_out   = acc;
    assign valid_out = valid_s2;

endmodule


module mem #(
    parameter DW = 8,
    parameter N  = 8
)(
    input  logic                   clk,
    input  logic                   rst_n,
    input  logic                   write_e,
    input  logic [$clog2(N*N)-1:0] waddr,
    input  logic [DW-1:0]          wdata,
    input  logic [$clog2(N*N)-1:0] raddr,
    output logic [DW-1:0]          rdata
);

    logic [DW-1:0] mem_array [0:N*N-1];

    always_ff @(posedge clk) begin
        if (write_e)
            mem_array[waddr] <= wdata;
    end

    assign rdata = mem_array[raddr];

endmodule


module postproc #(
    parameter int ACC_W = 32,
    parameter int SH_W  = 6
)(
    input  logic                    clk,
    input  logic                    rst_n,

    input  logic signed [ACC_W-1:0] acc,
    input  logic signed [ACC_W-1:0] bias,
    input  logic [SH_W-1:0]         shift,

    input  logic                    bias_en,
    input  logic                    relu_en,
    input  logic                    passthru,

    output logic signed [ACC_W-1:0] res32,
    output logic signed [7:0]       act8
);

    logic signed [ACC_W:0] biased_c;
    logic signed [ACC_W:0] half_c;
    logic signed [ACC_W:0] rnd_c;

    always_comb begin
        biased_c = $signed({acc[ACC_W-1], acc}) +
                   (bias_en ? $signed({bias[ACC_W-1], bias}) : '0);

        half_c = (shift == '0) ? '0 :
                 33'sd1 <<< (shift - 1);

        rnd_c = biased_c + half_c;
    end

    logic signed [ACC_W:0] biased_q;
    logic signed [ACC_W:0] rnd_q;
    logic [SH_W-1:0]       shift_q;
    logic                  relu_en_q;
    logic                  passthru_q;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            biased_q   <= '0;
            rnd_q      <= '0;
            shift_q    <= '0;
            relu_en_q  <= 1'b0;
            passthru_q <= 1'b0;
        end else begin
            biased_q   <= biased_c;
            rnd_q      <= rnd_c;
            shift_q    <= shift;
            relu_en_q  <= relu_en;
            passthru_q <= passthru;
        end
    end

    logic signed [ACC_W:0] shf;
    logic signed [ACC_W:0] relu;

    always_comb begin
        shf = rnd_q >>> shift_q;

        relu = (relu_en_q && (shf < 0)) ? '0 : shf;

        if (relu > 127)
            act8 = 8'sd127;
        else if (relu < -128)
            act8 = -8'sd128;
        else
            act8 = relu[7:0];

        res32 = passthru_q ? biased_q[ACC_W-1:0]
                           : {{(ACC_W-8){act8[7]}}, act8};
    end

endmodule
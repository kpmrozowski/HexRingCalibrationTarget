#include <filesystem>

#include <cmd_launcher/subcommand.hpp>
#include <io/save_path.hpp>

class CalibrateMono : public utils::Subcommand
{
   private:
    std::filesystem::path dataset_folder_;
    std::filesystem::path output_folder_;
    std::string camera_id_;
    std::vector<std::variant<int, float>> board_params_vec_;
    int board_type_ = 0;

   public:
    std::string name() const override { return "CalibrateMono"; }

    std::string description() const override { return "Calibrate mono camera"; }

    void set_options(CLI::App& cmd) override
    {
        add_dataset_path(cmd, dataset_folder_)->required();
        add_path_to_save(cmd, output_folder_)->default_val(io::save_path());
        add_camera(cmd, camera_id_);
        add_board_params(cmd, board_params_vec_);
        add_board(cmd, board_type_);
    }

    void execute() override;
};
